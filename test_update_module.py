import torch, cv2
from dpvo.net import VONet, CorrBlock
import numpy as np
from collections import OrderedDict
from dpvo.utils import flatmeshgrid
import os
import torch.nn.functional as F
from dpvo.lietorch import SE3
from dpvo import altcorr
from dpvo import projective_ops as pops

import matplotlib.pyplot as plt
autocast = torch.cuda.amp.autocast

def depth_read(depth_file):
    depth = np.load(depth_file) / 5.0
    depth[depth==np.nan] = 1.0
    depth[depth==np.inf] = 1.0
   
    return torch.from_numpy(depth).float().cuda()

def rgb_read(file):
    I = cv2.imread(file)
    return torch.from_numpy(I).permute(2,0,1).float().cuda()


state_dict = torch.load("dpvo.pth")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if "update.lmbda" not in k:
        new_state_dict[k.replace('module.', '')] = v

network = VONet()
network.load_state_dict(new_state_dict)

network.cuda()
network.eval()

base_path = "/home/zosurban/Projects/DPVO/datasets/TartanAir/abandonedfactory/Easy/P001"


Id = SE3.Identity(1, device="cuda")
class SimpleDPVO:
    def __init__(self, network, ht=480, wd=640):
        self.load_weights(network)
        self.is_initialized = False
        
        self.motion_damping = 0.5
        ### state attributes ###
        self.tlist = []
        self.counter = 0

        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = 50     # PATCHES_PER_FRAME
        self.N = 100    # BUFFERS_SIZE
        self.ht = ht    # image height
        self.wd = wd    # image width
        # store images
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")
        self.images_ = torch.zeros(self.N, 3, self.ht, self.wd, dtype=torch.uint8, device="cpu")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")
        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.image_tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ht = ht // self.RES
        wd = wd // self.RES

        ### network attributes ###
        self.mem = 32
        self.kwargs = {"device": "cuda", "dtype": torch.float}
        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **self.kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **self.kwargs)
        self.imap_ = torch.zeros(self.mem, self.M, self.DIM, **self.kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **self.kwargs)

        self.pyramid = (self.fmap1_, self.fmap2_)
        self.net = torch.zeros(1, 0, self.DIM, **self.kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        
        self.poses_[:,6] = 1.0
        self.delta = {}

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        corr = self.corr(coords, indicies=(kk, jj))
        ctx = self.imap[:,kk % (self.M * self.mem)]
        net, (delta, weight, _) = \
            self.network.update(net, ctx, corr, None, ii, jj, kk)

        img_idx1 = ii[0]
        img_idx2 = jj[0]

        I1 = cv2.cvtColor(self.images_[img_idx1].permute(1,2,0).cpu().numpy(),cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(self.images_[img_idx2].permute(1,2,0).cpu().numpy(),cv2.COLOR_BGR2GRAY)

        conc = np.concatenate([I1,I2],1)
        plt.imshow(conc)
        pts1 = self.patches_[img_idx1,:,:2,self.P//2,self.P//2].cpu().numpy().squeeze()*self.RES
        pts2 = self.patches_[img_idx2,:,:2,self.P//2,self.P//2].cpu().numpy().squeeze()*self.RES
        pts1_in_2 = coords[0,:,:2,self.P//2,self.P//2].cpu().numpy().squeeze()*self.RES
        pts1_in_2_update = (pts1 + delta[0].cpu().detach().numpy().squeeze()*self.RES)
        for i in range(self.M):
            plt.plot(pts1[i,0],pts1[i,1],'r+')
            #plt.plot(pts2[i,0]+self.wd,pts2[i,1],'g+')
            # plt.plot(pts1_in_2[i,0]+self.wd,pts1_in_2[i,1],'c+')
            plt.plot(pts1_in_2_update[i,0]+self.wd,pts1_in_2_update[i,1],'m+')
        plt.show()
        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def __call__(self, tstamp, image, depth_init, intrinsics, image_tstamp_ns):
        """ track new frame """

        self.images_[self.n,...] = image
        image = 2 * (image[None,None] / 255.0) - 0.5
                
        fmap, gmap, imap, patches, _, clr = \
            self.network.patchify(image,
                patches_per_image=self.M, 
                gradient_bias=False, 
                return_color=True)

        ### update state attributes ###
        self.tstamps_[self.n] = self.counter
        self.image_tstamps_[self.n] = image_tstamp_ns
        self.intrinsics_[self.n] = intrinsics / self.RES
        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            # linear motion model for next pose
            P1 = SE3(self.poses_[self.n-1])
            P2 = SE3(self.poses_[self.n-2])
            
            xi = self.motion_damping * (P1 * P2.inv()).log()
            tvec_qvec = (SE3.exp(xi) * P1).data
            self.poses_[self.n] = tvec_qvec


        # get depth at img_pts 
        img_pts = patches[:,:,:2,self.P//2,self.P//2].long()
        depth_small = depth_init[::self.RES, ::self.RES]
        patches[:,:,2] = depth_small[img_pts[0,:,1].flatten(), img_pts[0,:,0].flatten()].view(1,self.M,1,1).repeat(1,1,3,3) 
        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M


            

intr = np.array([320.0, 320.0, 320.0, 240.0])
intr = torch.from_numpy(intr).unsqueeze(0).cuda()

vo = SimpleDPVO("dpvo.pth")
I1 = rgb_read(os.path.join(base_path,"image_left/000002_left.png"))
d1 = 1/depth_read(os.path.join(base_path,"depth_left/000002_left_depth.npy"))
vo(0, I1, d1, intr, 0)

I2 = rgb_read(os.path.join(base_path,"image_left/000003_left.png"))
d2 = 1/depth_read(os.path.join(base_path,"depth_left/000003_left_depth.npy"))
vo(1, I2, d2, intr, 1)

I2 = rgb_read(os.path.join(base_path,"image_left/000004_left.png"))
d2 = 1/depth_read(os.path.join(base_path,"depth_left/000004_left_depth.npy"))
vo(1, I2, d2, intr, 1)