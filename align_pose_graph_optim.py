
import open3d as o3d
import numpy as np
import os
from telemetry_converter import TelemetryImporter
from dpvo.lietorch import SE3

import torch
from utils import load_dataset
# import GPS data

global_enu_llh0 = 0


base_path = "/media/Data/Sparsenet/TestAlignment"
tel_importer = TelemetryImporter()
tel_importer.read_gopro_telemetry(os.path.join(base_path,"bike1_trail1_linear.json"))
llh01 = tel_importer.telemetry["gps_llh"][0]
tel_importer = TelemetryImporter()
tel_importer.read_gopro_telemetry(os.path.join(base_path,"bike2_trail1_linear.json"))
llh02 = tel_importer.telemetry["gps_llh"][0]

p_w_c1, q_w_c1, grav1, gps1, t_ns1 = load_dataset(
    os.path.join(base_path,"dpvo_result_bike1_trail1_linear.npz"),
    os.path.join(base_path,"bike1_trail1_linear.json"),
    llh01)

p_w_c2, q_w_c2, grav2, gps2, t_ns2 = load_dataset(
    os.path.join(base_path,"dpvo_result_bike2_trail1_linear.npz"),
    os.path.join(base_path,"bike2_trail1_linear.json"),
    llh02)

# read relative constrain    

import gtsam
import gtsam.utils.plot as gtsam_plot


vtx_cnt = 1

# add poses and odometry constrains

def SE3_from_t_q(t,q):
    return SE3(torch.tensor(np.array([t[0],t[1],t[2],q[0],q[1],q[2],q[3]])))

# Create a Nonlinear factor graph as well as the data structure to hold state estimates.
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-5,1e-5,1e-5,1e-5,1e-5,1e-5]))

ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.,1.,1.,1.,1.,1.]))

GT_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]))


# Add the prior factor to the factor graph, and poorly initialize the prior pose to demonstrate
# iSAM2 incremental optimization.
pose1 = gtsam.Pose3(SE3_from_t_q(p_w_c1[0,:],q_w_c1[0,:]).matrix())
graph.push_back(gtsam.PriorFactorPose3(vtx_cnt, pose1, PRIOR_NOISE))
initial_estimate.insert(vtx_cnt, pose1)
    
# add first trajectory
t_ns1_to_vertex_idx = {t_ns1[0] : vtx_cnt}
for i in range(1,p_w_c1.shape[0]):
    old_pose = SE3_from_t_q(p_w_c1[i-1,:],q_w_c1[i-1,:])
    cur_pose = SE3_from_t_q(p_w_c1[i,:],q_w_c1[i,:])
    odometry_se3 = cur_pose * old_pose.inv()
    odometry_se3_mat = odometry_se3.matrix()
    graph.push_back(gtsam.BetweenFactorPose3(vtx_cnt, vtx_cnt+1, gtsam.Pose3(odometry_se3_mat), ODOMETRY_NOISE))
    initial_estimate.insert(vtx_cnt+1, gtsam.Pose3(cur_pose.matrix()))
    graph.push_back(gtsam.PriorFactorPose3(vtx_cnt, gtsam.Pose3(cur_pose.matrix()), PRIOR_NOISE))

    vtx_cnt += 1
    t_ns1_to_vertex_idx[t_ns1[i]] = vtx_cnt

vtx_cnt += 1
pose1 = gtsam.Pose3(SE3_from_t_q(p_w_c2[0,:],q_w_c2[0,:]).matrix())
initial_estimate.insert(vtx_cnt, pose1)
t_ns2_to_vertex_idx = {t_ns2[0] : vtx_cnt}
# add second trajectory
for i in range(1,p_w_c2.shape[0]):
    old_pose = SE3_from_t_q(p_w_c2[i-1,:],q_w_c2[i-1,:])
    cur_pose = SE3_from_t_q(p_w_c2[i,:],q_w_c2[i,:])
    odometry_se3 = cur_pose * old_pose.inv()
    odometry_se3_mat = odometry_se3.matrix()
    graph.push_back(gtsam.BetweenFactorPose3(vtx_cnt, vtx_cnt+1, gtsam.Pose3(odometry_se3_mat), ODOMETRY_NOISE))
    initial_estimate.insert(vtx_cnt+1, gtsam.Pose3(cur_pose.matrix()))
    vtx_cnt += 1
    t_ns2_to_vertex_idx[t_ns2[i]] = vtx_cnt

# add loop closure constrain
# read matching keyframe timestamps csv file
import csv
timestamp_pairs = []
with open(os.path.join(base_path,"maching_kf_tns.txt"), "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        timestamp_pairs.append(row)
rel_pose_constrains = np.load(os.path.join(base_path,"relative_trafos.txt.npy"))

vtx1 = t_ns1_to_vertex_idx[int(timestamp_pairs[0][0])]
vtx2 = t_ns2_to_vertex_idx[int(timestamp_pairs[0][1])]

graph.push_back(gtsam.BetweenFactorPose3(vtx1, vtx2, gtsam.Pose3(rel_pose_constrains), GT_ODOMETRY_NOISE))

params = gtsam.GaussNewtonParams()
params.setVerbosity(
    "Termination")  # this will show info about stopping conds
optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()
print("Optimization complete")

from gtsam.utils import plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

resultPoses = gtsam.utilities.allPose3s(result)

traj1_ts = []
traj2_ts = []

for i in range(resultPoses.size()):
    t = resultPoses.atPose3(i+1).translation()
    if i+1 > vtx1:
        traj2_ts.append(t.tolist())
    else:
        traj1_ts.append(t.tolist())



pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(np.asarray(traj1_ts))
pcl1.paint_uniform_color([1, 0, 0])
pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector(np.asarray(traj2_ts))
pcl2.paint_uniform_color([0, 1,  0])

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=600, height=500, left=450, top=250)
visualizer.add_geometry(pcl1)
visualizer.add_geometry(pcl2)
visualizer.add_geometry(world_frame)

view_ctl = visualizer.get_view_control()
view_ctl.set_front((1, 0, 0))
view_ctl.set_up((0, 0, 1))  # can't set z-axis as the up-axis
view_ctl.set_lookat((0, 0, 0))

visualizer.run()
visualizer.destroy_window()
