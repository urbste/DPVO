from venv import create
import open3d as o3d
import pyvisfm as pvi
import os
from utils import load_dataset
import numpy as np
from scipy.spatial.transform import Rotation as R 

base_path = "/media/Data/Sparsenet/Ammerbach/Links"

def create_pcl(coords, color):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(coords)
    pcl.paint_uniform_color(color)

    return pcl

# load VO dataset
dataset = load_dataset(
    os.path.join(base_path,"dpvo_result_run1.npz"),
    os.path.join(base_path,"run1.json"),
    None, inv_depth_thresh=0.5, 
    scale_with_gps=False, align_with_grav=False, correct_heading=True)

kf_t_ns = dataset["frametimes_slam_ns"]
gps_ned_coords = np.array(dataset["gps_local_ned"])

spline = pvi.SplineTrajectoryEstimator()
pvi.ReadSpline(spline, os.path.join(base_path,"spline_recon_run1.spline"))

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])

p_w_c, q_w_c, gps_pos = [], [], []
for idx, t_ns in enumerate(kf_t_ns):
    T_w_c = spline.GetCameraPose(t_ns)
    p_w_c.append(T_w_c[4:])
    q_w_c.append(R.from_quat(T_w_c[0:4]).as_quat())
    gps_pos.append(gps_ned_coords[idx,:]-gps_ned_coords[0,:])
p_w_c = np.array(p_w_c)
q_w_c = np.array(q_w_c)
gps_pos = np.array(gps_pos)
cam_spline = create_pcl(p_w_c, [1,0,0])
gps_world = create_pcl(gps_pos, [0,1,0])

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=1000, height=1000, left=450, top=250)
opt = visualizer.get_render_option()
opt.background_color = np.asarray([0.8, 0.8, 0.8])
visualizer.add_geometry(cam_spline)
visualizer.add_geometry(gps_world)

for i in range(p_w_c.shape[0]):
    T = np.eye(4)
    T[:3,:3] = R.from_quat(q_w_c[i,:]).as_matrix()
    T[:3,3] = p_w_c[i,:]
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
    visualizer.add_geometry(cam_frame.transform(T))

#visualizer.add_geometry(mesh1)
#visualizer.add_geometry(map_pts_pcl2)
#visualizer.add_geometry(gps_pcl1)
#visualizer.add_geometry(gps_pcl2)
visualizer.add_geometry(world_frame)

view_ctl = visualizer.get_view_control()
view_ctl.set_front((1, 0, 0))
view_ctl.set_up((0, 0, 1))  # can't set z-axis as the up-axis
view_ctl.set_lookat((0, 0, 0))

visualizer.run()
visualizer.destroy_window()


