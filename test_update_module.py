import torch, cv2
from dpvo.net import VONet, CorrBlock
import numpy as np
from collections import OrderedDict
from dpvo.utils import flatmeshgrid

def depth_read(depth_file):
    depth = np.load(depth_file) / 5.0
    depth[depth==np.nan] = 1.0
    depth[depth==np.inf] = 1.0
    return depth[::2,::2]

def rgb_read(file):
    I = cv2.imread(file)
    I = cv2.resize(I, (320, 240))
    return I


state_dict = torch.load("dpvo.pth")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if "update.lmbda" not in k:
        new_state_dict[k.replace('module.', '')] = v

network = VONet()
network.load_state_dict(new_state_dict)

network.cuda()
network.eval()

# load two images
images = [rgb_read("/home/steffen/projects/DPVO/datasets/TartanAir/abandonedfactory/abandonedfactory/Easy/P000/image_left/000002_left.png"), 
          rgb_read("/home/steffen/projects/DPVO/datasets/TartanAir/abandonedfactory/abandonedfactory/Easy/P000/image_left/000003_left.png")]
intr = np.array([320.0, 320.0, 320.0, 240.0])/2
intrinsics = [intr] * len(images)
depths = [depth_read("/home/steffen/projects/DPVO/datasets/TartanAir/abandonedfactory/abandonedfactory/Easy/P000/depth_left/000002_left_depth.npy"),
          depth_read("/home/steffen/projects/DPVO/datasets/TartanAir/abandonedfactory/abandonedfactory/Easy/P000/depth_left/000003_left_depth.npy")]

images = np.stack(images).astype(np.float32)
depths = np.stack(depths).astype(np.float32)
intrinsics = np.stack(intrinsics).astype(np.float32)

images = torch.from_numpy(images).float().cuda()
images = images.permute(0, 3, 1, 2).unsqueeze(0)
disps = torch.from_numpy(1.0 / depths).unsqueeze(0).cuda()
intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).cuda()

images = 2 * (images / 255.0) - 0.5
intrinsics = intrinsics / 4.0
disps = disps[:, :, 1::4, 1::4].float()

fmap, gmap, imap, patches, ix = network.patchify(images, disps=disps)
corr_fn = CorrBlock(fmap, gmap)

kk, jj = flatmeshgrid(torch.where(ix < 2)[0], torch.arange(0,1, device="cuda"))
ii = ix[kk]

patches_gt = patches.clone()
#Ps = poses
# depth of patch at center
p=3
d = patches[..., 2, p//2, p//2]
pts2 = patches[...,:2, p//2, p//2].cpu().numpy()
#patches = set_depth(patches, torch.rand_like(d))


import matplotlib.pyplot as plt
_, axs = plt.subplots(2,1)
axs[0].imshow(images[0,0,0].cpu().numpy())
axs[1].imshow(images[0,1,0].cpu().numpy())

for idx, i in enumerate(ii.cpu().numpy().tolist()):
    axs[i].plot(pts2[0,idx,0]*4,pts2[0,idx,1]*4,'r+')
plt.show()

# now project coords to next frame and calculate correlation features 
# then run update operator
from dpvo import projective_ops as pops
from dpvo.lietorch import SE3
from dpvo.projective_ops import iproj, proj
Gs = SE3.IdentityLike(torch.tensor([1,2]).unsqueeze(0))
Gij = Gs[:, jj] * Gs[:, ii].inv()
bearings = Gij[:,:,None,None] * iproj(patches[:,kk],intrinsics[:,ii])
coords = proj(bearings, intrinsics[:,jj], False)
# coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

corr = corr_fn(kk, jj, coords1)
net = torch.zeros(1, len(kk), 384, device="cuda", dtype=torch.float)
net, (delta, weight, _) = network.update(net, imap[:,kk], corr, None, ii, jj, kk)

print()