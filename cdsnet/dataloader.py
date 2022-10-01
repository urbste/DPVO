import os, cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from natsort import natsorted

def mvs_loader(image_dir, cam_mat, tstamps, poses, min_max_d):

    h1, w1 = 512, 896
    images = []
    transform = Compose([Resize((h1, w1)), ToTensor()])
    h0, w0 = None, None
    for t in tstamps:
        imfile = str(t) + ".png"
        # image = cv2.imread(os.path.join(args.imagedir, imfile))
        # if len(calib) > 4:
        #     image = cv2.undistort(image, K, calib[4:])
        image = Image.open(os.path.join(image_dir, imfile))
        w0, h0 = image.size[:2]

        image = transform(image) #cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]

        # image = torch.as_tensor(image).permute(2, 0, 1)
        # image = image.float() / 255.
        images.append(image)
    fx, cx = cam_mat[0,0] * (w1 / w0), cam_mat[0,2] * (w1 / w0)
    fy, cy = cam_mat[1,1] * (h1 / h0), cam_mat[1,2] * (h1 / h0)
    images = torch.stack(images, dim=0)

    intr_mat = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=poses.device)

    intr_matrices = intr_mat.unsqueeze(0).repeat(poses.size(0), 1, 1)
    proj_stage3 = torch.stack((poses, intr_matrices), dim=1)
    proj_stage2 = proj_stage3.clone()
    proj_stage2[:, 1, :2] *= 0.5
    proj_stage1 = proj_stage2.clone()
    proj_stage1[:, 1, :2] *= 0.5
    proj_stage0 = proj_stage1.clone()
    proj_stage0[:, 1, :2] *= 0.5
    proj_matrices = {"stage1": proj_stage0.unsqueeze(0).cuda(),
                     "stage2": proj_stage1.unsqueeze(0).cuda(),
                     "stage3": proj_stage2.unsqueeze(0).cuda(),
                     "stage4": proj_stage3.unsqueeze(0).cuda()}

    min_d = min_max_d[0]
    max_d = min_max_d[1]
    d_interval = (max_d - min_d) / 512
    depth_values = torch.arange(0, 512, dtype=torch.float32).unsqueeze(0) * d_interval + min_d
    return images.unsqueeze(0).cuda(), proj_matrices, depth_values.cuda()
