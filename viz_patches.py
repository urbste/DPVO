import open3d as o3d
import numpy as np
from dpvo.lietorch import SE3
import torch


def load_dataset(path):

    data = np.load(path)
    poses_w_c = data["name2"]
    num_kfs = poses_w_c.shape[0]
    frametimes_slam_ns = data["name4"].astype(np.int64)[:num_kfs]
    frame_ids = data["name3"][:num_kfs]
    patches = data["name5"][:num_kfs,...]
    ii, jj, kk = data["name7"], data["name8"], data["name9"]
    intr = data["name10"]

    se3_poses = SE3(torch.tensor(poses_w_c).unsqueeze(0))
    p_w_c = se3_poses.translation()[:,0:3]
    q_w_c = se3_poses.data[:,3:]

    return p_w_c.numpy(), q_w_c.numpy(), poses_w_c, patches, ii, jj, kk, frametimes_slam_ns, intr


p, q, se3_poses, patches, ii, jj, kk, t_ns, intr = load_dataset("/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/dpvo_result_bike1_trail1_linear.npz")

import cv2

start_id = 40

win_size = 5
# read some frames to test stuff
ti = t_ns[start_id]
tj = t_ns[start_id+1]

pi = patches[start_id]
num_patches_per_img = patches.shape[1]
total_num_patches = patches.shape[0]*patches.shape[1]
flattened_patches = torch.tensor(patches).view(1, total_num_patches, 3, 3, 3)

Ii = cv2.imread("/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_linear_imgs/"+str(t_ns[start_id])+".png")

# plot 64 points
def plot_points(img, img_pts):
    for i in range(img_pts.shape[0]):
        p3 = img_pts[i][:,1,1]
        int_p2 = (int(p3[0]*4), int(p3[1]*4))
        img = cv2.drawMarker(img, int_p2, (0,0,255), cv2.MARKER_SQUARE, 8, 1)
        img = cv2.putText(img, str(i), int_p2, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0),1)
    return img



from dpvo import projective_ops as pops
def reproject(ii, jj, kk, patches, intrinsics, poses):

    coords = pops.transform(poses, patches, intrinsics, ii, jj, kk)
    return coords.permute(0, 1, 4, 2, 3).contiguous()


I_i_w_pts = plot_points(Ii, pi)

cv2.imshow("Ii", I_i_w_pts)
cv2.waitKey(0)

# now project to next frames
for idx, t in enumerate(t_ns[start_id-win_size:start_id+win_size]):
    Ij = cv2.imread("/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_linear_imgs/"+str(t)+".png")

    i = torch.zeros((num_patches_per_img)) + start_id
    j = i + (idx - win_size)
    k = torch.arange(0,num_patches_per_img) + i * num_patches_per_img
    coords = reproject(
            i.long(), j.long(), k.long(), 
            flattened_patches, 
            torch.tensor(intr).unsqueeze(0), 
            SE3(torch.tensor(np.expand_dims(se3_poses,0))).inv())
    
    I_j_w_pts = plot_points(Ij, coords.squeeze(0))

    cv2.imshow("projected_patches",np.concatenate([I_i_w_pts, I_j_w_pts],0))
    cv2.waitKey(0)