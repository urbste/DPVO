import os
import natsort
from collections import OrderedDict

from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from cdsnet.cdsmvsnet import CDSMVSNet
from cdsnet.dataloader import mvs_loader
import torch.nn.functional as F
from telemetry_converter import TelemetryImporter
import pyvisfm as pvi

from utils import load_dataset

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]]) / 3

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_point_actor(rgbd_image, intr):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intr)
    return point_cloud

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def get_cam_pose_from_spline_at_time(spline, time_ns):
    pose_vec = spline.GetCameraPose(int(time_ns))
    R_w_c = R.from_quat([pose_vec[0],pose_vec[1],pose_vec[2],pose_vec[3]]).as_matrix()
    p_w_c = np.expand_dims(pose_vec[4:],1)
    R_c_w = R_w_c.T
    t_c_w = -R_c_w@p_w_c
    T_c_w = np.eye(4)
    T_c_w[:3,:3] = R_c_w
    T_c_w[:3,3] = t_c_w.squeeze()
    return R_c_w, t_c_w, p_w_c, T_c_w


mvsnet = CDSMVSNet(refine=True, ndepths=(128, 32, 8), depth_interals_ratio=(4, 4, 1))
mvsnet_ckpt = torch.load("cds_mvsnet.pth")
state_dict = OrderedDict([
    (k.replace("module.", ""), v) for (k, v) in mvsnet_ckpt["state_dict"].items()
])
mvsnet.load_state_dict(state_dict, strict=False)
mvsnet.to("cuda:0").eval()


base_path = "/media/Data/Sparsenet/Ammerbach/Links"
run = "run1"
path_traj1 = os.path.join(base_path,run)
tel_json = run+".json"
dpvo_res = "dpvo_result_"+run+".npz"
dataset = load_dataset(
    os.path.join(base_path,dpvo_res),
    os.path.join(base_path,tel_json),
    None, inv_depth_thresh=0.5, 
    scale_with_gps=True, align_with_grav=True, correct_heading=True)
kf_timstamps = dataset["frametimes_slam_ns"]


imsize = (640, 480)
cam_mat = np.diag([297.4347120333558, 297.4347120333558, 1.]) 
cam_mat[0,2] =  323.609635667
cam_mat[1,2] =  237.52771880186

spline = pvi.SplineTrajectoryEstimator()
pvi.ReadSpline(spline, 
    os.path.join(base_path,"spline_recon_"+run+".spline"))

# load VO dataset
dataset = load_dataset(
    os.path.join(base_path,dpvo_res),
    os.path.join(base_path,tel_json),
    None, inv_depth_thresh=0.5, 
    scale_with_gps=True, align_with_grav=True, correct_heading=True)

# generate image poses
image_folder = os.listdir(path_traj1)

images = []
depth =[]

spline_ = {}
p_w_c = dataset["p_w_c"]
q_w_c =dataset["q_w_c"]
for idx, t_ns in enumerate(kf_timstamps):
    R_w_c = R.from_quat(q_w_c[idx]).as_matrix()
    
    T_c_w = np.eye(4)
    T_c_w[:3,:3] = R_w_c.T
    T_c_w[:3,3] = -R_w_c.T@np.array(p_w_c[idx])
    #spline_[t_ns] = T_c_w # get_cam_pose_from_spline_at_time(spline, int(t_ns))[3]
    spline_[t_ns] = get_cam_pose_from_spline_at_time(spline, int(t_ns))[3]
import open3d as o3d
import open3d.core as o3c
import time

# vbg = o3d.t.geometry.VoxelBlockGrid(
#     attr_names=('tsdf', 'weight', 'color'),
#     attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
#     attr_channels=((1), (1), (3)),
#     voxel_size=3.0 / 512,
#     block_resolution=16,
#     block_count=50000,
#     device="cuda")

vis = o3d.visualization.Visualizer()
vis.create_window()
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for t in range(2, 20):
    print("Estimating depth for KF {}",t)
    tstamps = [kf_timstamps[t],kf_timstamps[t-1], kf_timstamps[t+1]]
    poses = torch.zeros(3, 4, 4).float()
    for idx, p in enumerate(tstamps):
        poses[idx,:,:] = torch.tensor(spline_[p])
    images, proj_matrices, depth_values = mvs_loader(path_traj1, cam_mat, tstamps, poses, [0.5, 20.])

    with torch.no_grad():
        mvs_outputs = mvsnet(images.cuda(), proj_matrices, depth_values.cuda(), temperature=0.01)
        final_depth = mvs_outputs["refined_depth"]
        mask = torch.ones_like(final_depth) > 0.0
        for stage, thresh_conf in zip(["stage1", "stage2", "stage3"], [0.7, 0.8, 0.9]):
            conf_stage = F.interpolate(mvs_outputs[stage]["photometric_confidence"].unsqueeze(1),
                                        (mask.size(1), mask.size(2))).squeeze(1)
            mask = mask & (conf_stage > thresh_conf)
        final_depth[~mask] = 1000

        final_depth = final_depth.squeeze(0)
        _, (ax1, ax2, ax3) = plt.subplots(3,1)
        ax1.imshow(final_depth.squeeze(0).cpu().numpy(), cmap='jet', vmin = 1e-6,vmax = 20.)
        ax2.imshow(images.squeeze(0)[0,0,...].cpu().numpy())
        ax3.imshow(conf_stage.squeeze(0).cpu().numpy())

        plt.show()

    color = np.ascontiguousarray(images.squeeze(0)[0,...].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    depth_img = np.ascontiguousarray(final_depth.squeeze(0).cpu().numpy()).astype(np.float32)

    #cv2.imwrite("depth_img.png", (depth_img*10000).astype(np.uint16))
    #cv2.imwrite("color.png", (color*255).astype(np.uint8))

    intr = proj_matrices['stage4'][0,0,1,:3,:3].cpu().numpy()
    intr_o3d = o3d.core.Tensor(intr, o3d.core.Dtype.Float64)
    #images.append(images.squeeze(0)[0,...].permute(1, 2, 0).cpu().numpy())
    #depth.append(final_depth.squeeze(0).cpu().numpy())

    rgbd_intr = o3d.camera.PinholeCameraIntrinsic()
    rgbd_intr.set_intrinsics(color.shape[1], color.shape[0], intr[0,0], intr[1,1], intr[0,2], intr[1,2])
    #d_o3d = o3d.io.read_image("depth_img.png")
    #c_o3d = o3d.io.read_image("color.png")
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth_img), depth_scale=1.0, 
            depth_trunc=10.0, convert_rgb_to_intensity=False)

    point_actor = create_point_actor(rgbd, rgbd_intr)
    volume.integrate(rgbd, rgbd_intr, spline_[kf_timstamps[t]].astype(np.float64))
    T_w_c = np.linalg.inv(spline_[kf_timstamps[t]].astype(np.float64))
    point_actor.transform(T_w_c)
    cam_actor = create_camera_actor(True)
    cam_actor.transform(T_w_c)

    vis.add_geometry(point_actor)
    vis.add_geometry(cam_actor)


    # extr_o3d = o3d.core.Tensor(spline_[t], o3d.core.Dtype.Float64)
    # intr_o3d = o3d.core.Tensor(proj_matrices['stage4'][0,0,1,...].cpu().numpy(), o3d.core.Dtype.Float64)

    # depth_img_o3d = final_depth.squeeze(0).cpu().numpy()
    # frustum_block_coords = vbg.compute_unique_block_coordinates(
    #     depth_img_o3d, intr_o3d, extr_o3d, 1.0,20.)
    # color_img_o3d = images.squeeze(0)[0,...].permute(1, 2, 0).cpu().numpy()
    # vbg.integrate(frustum_block_coords, depth, color_img_o3d,
    #                     intr_o3d, intr_o3d, extr_o3d,
    #                     1.0, 20.0)
vis.run()

mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
