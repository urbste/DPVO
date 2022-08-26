import open3d as o3d
import numpy as np
from dpvo.lietorch import SE3
import torch
import os
from telemetry_converter import TelemetryImporter
from natsort import natsorted
from gps_converter import ECEFtoENU
# import GPS data

global_enu_llh0 = 0

def load_dataset(path, telemetry_file, llh0):

    data = np.load(path)
    poses_w_c = data["name2"]
    num_kfs = poses_w_c.shape[0]
    frametimes_slam_ns = data["name4"].astype(np.int64)[:num_kfs]
    frame_ids = data["name3"][:num_kfs]
    patches = data["name5"][:num_kfs,...]
    ii, jj, kk = data["name7"], data["name8"], data["name9"]

    tel_importer = TelemetryImporter()
    tel_importer.read_gopro_telemetry(telemetry_file)
    gps_xyz, _ = tel_importer.get_gps_pos_at_frametimes(frametimes_slam_ns)
    gravity_vectors = tel_importer.get_gravity_vector_at_times(frametimes_slam_ns)

    p_w_c = SE3(torch.tensor(poses_w_c)).translation()[:,0:3]
    q_w_c = SE3(torch.tensor(poses_w_c)).data[:,3:]

    gps_enu_at_kfs = [ECEFtoENU(gps_xyz[int(key)], llh0) if int(key) in gps_xyz else print(key) for key in frametimes_slam_ns]
    
    return p_w_c.numpy(), q_w_c.numpy(), gravity_vectors, np.array(gps_enu_at_kfs, dtype=np.float32)

tel_importer = TelemetryImporter()
tel_importer.read_gopro_telemetry("/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_linear.json")
llh01 = tel_importer.telemetry["gps_llh"][0]
tel_importer = TelemetryImporter()
tel_importer.read_gopro_telemetry("/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike2_trail1_linear.json")
llh02 = tel_importer.telemetry["gps_llh"][0]

p_w_c1, q_w_c1, grav1, gps1 = load_dataset(
    "/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/dpvo_result_bike1_trail1_linear.npz",
    "/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_linear.json",
    llh01)

p_w_c2, q_w_c2, grav2, gps2 = load_dataset(
    "/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/dpvo_result_bike2_trail1_linear.npz",
    "/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike2_trail1_linear.json",
    llh02)


# find rotation to world frame for first trajectory
def rot_between_vectors(a,b):
    # rotates a -> b
    def skew(vector):
        return np.array([[0, -vector[2], vector[1]], 
                        [vector[2], 0, -vector[0]], 
                        [-vector[1], vector[0], 0]])

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a,b)
    c = np.dot(a,b)
    s = np.linalg.norm(v)

    R = np.eye(3) + skew(v) + np.linalg.matrix_power(skew(v),2)*((1-c)/s**2)

    return R

from scipy.spatial.transform import Rotation as R
R_imu_to_cam = R.from_quat([-0.0002947483347789445,-0.7104874521708413,0.7037096975195423,0.000393761552164708]).as_matrix().T

def get_rot_to_worldframe(gravity_vecors, q_w_c, world_vec=np.array([0,0,-1])):
    mean_vec = []
    for i in range(gravity_vecors.shape[0]):
        R_cam_to_world = rot_between_vectors(gravity_vecors[i], world_vec)
        Rij = R_cam_to_world @ R.from_quat(q_w_c[i,:]).as_matrix().T
        mean_vec.append(R.from_matrix(Rij).as_rotvec())

    return R.from_rotvec(np.median(mean_vec,0)).as_matrix()

R1_to_grav = get_rot_to_worldframe(grav1, q_w_c1)
R2_to_grav = get_rot_to_worldframe(grav2, q_w_c2)


def get_heading_rot(position_data):
    traj_direction = position_data[-1]- position_data[0]
    traj_direction[2] = 0.0
    traj_direction /= np.linalg.norm(traj_direction)

    angle_to_y_axis = np.arcsin(traj_direction[0]/np.sqrt(
        traj_direction[1]**2 + traj_direction[0]**2))
    return R.from_rotvec([0,0,angle_to_y_axis]).as_matrix()

def get_vis_scaler(vis_pos, gps_pos):
    # scale trajectory with gps first and last pose
    d_vis = np.linalg.norm(vis_pos[-1]- vis_pos[0])
    d_gps = np.linalg.norm(gps_pos[-1]- gps_pos[0])
    return d_gps/d_vis

def norm_vec(vec):
    return vec / np.linalg.norm(vec)

def get_heading_angle_diff(norm_vis, norm_gps):
    # angle between vectors aroun z-axis
    dir_gps = norm_vec(norm_gps[-1] - norm_gps[0])
    dir_vis = norm_vec(norm_vis[-1] - norm_vis[0])
    return np.arccos(np.dot(dir_gps[:2],dir_vis[:2]))


s1 = get_vis_scaler(p_w_c1, gps1)
s2 = get_vis_scaler(p_w_c2, gps2)

norm_vis1 = (s1*R1_to_grav@p_w_c1.T).T
norm_vis2 = (s2*R2_to_grav@p_w_c2.T).T

norm_gps1 = gps1-gps1[0]
norm_gps2 = gps2-gps2[0]


heading1 = R.from_rotvec([0,0,-get_heading_angle_diff(norm_vis1, norm_gps1)]).as_matrix()
heading2 = R.from_rotvec([0,0,-get_heading_angle_diff(norm_vis2, norm_gps2)]).as_matrix()


pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector((heading1@norm_vis1.T).T)
pcl1.paint_uniform_color([1, 0.706, 1])
pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector((heading2@norm_vis2.T).T)
pcl2.paint_uniform_color([1, 1,  0.706])

gps_pcl1 = o3d.geometry.PointCloud()
gps_pcl1.points = o3d.utility.Vector3dVector(gps1-gps1[0])
gps_pcl1.paint_uniform_color([1, 0.706, 0])
gps_pcl2 = o3d.geometry.PointCloud()
gps_pcl2.points = o3d.utility.Vector3dVector(gps2-gps2[0])
gps_pcl2.paint_uniform_color([1, 0,  0.706])


world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=600, height=500, left=450, top=250)
visualizer.add_geometry(pcl1)
visualizer.add_geometry(pcl2)
visualizer.add_geometry(gps_pcl1)
visualizer.add_geometry(gps_pcl2)
visualizer.add_geometry(world_frame)

view_ctl = visualizer.get_view_control()
view_ctl.set_front((1, 0, 0))
view_ctl.set_up((0, 0, 1))  # can't set z-axis as the up-axis
view_ctl.set_lookat((0, 0, 0))

visualizer.run()
visualizer.destroy_window()
