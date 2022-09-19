
import open3d as o3d
import numpy as np
import os
from telemetry_converter import TelemetryImporter
from dpvo.lietorch import SE3
from scipy.spatial.transform import Rotation as R
from gps_converter import ECEFToLLA
import pickle
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

# add poses and odometry constrains

def SE3_from_t_q(t,q):
    return SE3(torch.tensor(np.array([t[0],t[1],t[2],q[0],q[1],q[2],q[3]])))

def Pose3_from_SE3(se3):
    return gtsam.Pose3(se3.matrix())

# Create a Nonlinear factor graph as well as the data structure to hold state estimates.
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1]))

ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.,1.,1.,1.,1.,1.]))

GT_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]))

GPS_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([5, 5, 10]))

# read matching keyframe timestamps csv file
rel_constr_dict = pickle.load(open(os.path.join(base_path,"relative_trafos.dict"),"rb"))
vtx_cnt = 1
# add prior poses from trajectory 1
t_ns1_to_vertex_idx = {}
# for t_ns_1 in rel_constr_dict.keys():
#     idx_pose = list(t_ns1).index(t_ns_1)

#     # Add the prior factor to the factor graph, and poorly initialize the prior pose to demonstrate
#     # iSAM2 incremental optimization.
#     pose1 = gtsam.Pose3(SE3_from_t_q(p_w_c1[idx_pose,:],q_w_c1[idx_pose,:]).matrix())
#     graph.push_back(gtsam.PriorFactorPose3(vtx_cnt, pose1, PRIOR_NOISE))
#     graph.push_back(gtsam.GPSFactor(vtx_cnt,gtsam.Point3(ECEFToLLA(gps1[idx_pose,:]))))
#     initial_estimate.insert(vtx_cnt, pose1)
    
#     t_ns1_to_vertex_idx[t_ns_1] = vtx_cnt
#     vtx_cnt += 1

for i in range(0,p_w_c1.shape[0]):
    # Add the prior factor to the factor graph, and poorly initialize the prior pose to demonstrate
    # iSAM2 incremental optimization.
    pose1 = gtsam.Pose3(SE3_from_t_q(p_w_c1[i,:],q_w_c1[i,:]).matrix())
    graph.push_back(gtsam.PriorFactorPose3(vtx_cnt, pose1, PRIOR_NOISE))
    
    # graph.push_back(gtsam.PoseTranslationPrio(vtx_cnt)
    #graph.push_back(gtsam.GPSFactor(vtx_cnt,gtsam.Point3(ECEFToLLA(gps1[i,:])),GPS_NOISE))

    # attitude factor
    #graph.push_back()

    initial_estimate.insert(vtx_cnt, pose1)
    
    t_ns1_to_vertex_idx[t_ns1[i]] = vtx_cnt
    vtx_cnt += 1

# now add entire trajectory 2 and add loop closure constrains
pose1 = gtsam.Pose3(SE3_from_t_q(p_w_c2[0,:],q_w_c2[0,:]).matrix())
initial_estimate.insert(vtx_cnt, pose1)
t_ns2_to_vertex_idx = {t_ns2[0] : vtx_cnt}
# add second trajectory
for i in range(1,p_w_c2.shape[0]):
    old_pose = Pose3_from_SE3(SE3_from_t_q(p_w_c2[i-1,:],q_w_c2[i-1,:]))
    cur_pose = Pose3_from_SE3(SE3_from_t_q(p_w_c2[i,:],q_w_c2[i,:]))

    odometry = old_pose.between(cur_pose)

    graph.push_back(gtsam.BetweenFactorPose3(vtx_cnt, vtx_cnt+1, odometry, ODOMETRY_NOISE))
    initial_estimate.insert(vtx_cnt+1, gtsam.Pose3(cur_pose.matrix()))
    t_ns2_to_vertex_idx[t_ns2[i]] = vtx_cnt+1
    vtx_cnt += 1

for t_ns_1 in rel_constr_dict:
    vtx1 = t_ns1_to_vertex_idx[t_ns_1]
    t_ns_2 = rel_constr_dict[t_ns_1]["t_ns_2"]
    vtx2 = t_ns2_to_vertex_idx[t_ns_2]

    T01 = np.eye(4)
    T01[:3,:3] = R.from_quat(rel_constr_dict[t_ns_1]["q01"]).as_matrix()
    T01[:3,3] = np.array(rel_constr_dict[t_ns_1]["t01"])

    graph.push_back(gtsam.BetweenFactorPose3(vtx1, vtx2, gtsam.Pose3(T01), GT_ODOMETRY_NOISE))

params = gtsam.LevenbergMarquardtParams()
params.setVerbosity(
    "TERMINATION")  # this will show info about stopping conds
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()
print("Pose Graph Optimization complete")

from gtsam.utils import plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

resultPoses = gtsam.utilities.allPose3s(result)

traj1_ts, traj2_ts = [], []
traj1_qs, traj2_qs = [], []
for i in range(1,resultPoses.size()):
    t = resultPoses.atPose3(i).translation()
    q = R.from_matrix(resultPoses.atPose3(i).matrix()[:3,:3]).as_quat()
    if i > vtx1:
        traj2_ts.append(t.tolist())
        traj2_qs.append(q.tolist())
    else:
        traj1_ts.append(t.tolist())
        traj1_qs.append(q.tolist())

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

# # scale trajectory
# scale_gps = np.linalg.norm(gps1[-1] - gps1[0])
# scale_vis = np.linalg.norm(p_w_c1[-1] - p_w_c1[0])
# scale = scale_gps / scale_vis

# transform traj to gps space
gps_pcl = o3d.geometry.PointCloud()
gps_pcl.points = o3d.utility.Vector3dVector(np.asarray(gps1))
gps_pcl.paint_uniform_color([0, 0, 1])

n_pts = gps1.shape[0]
ids = o3d.utility.Vector2iVector(np.array([np.arange(0,n_pts),np.arange(0,n_pts)]).T)

trafo_est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=1)
T_gps_cam = trafo_est.compute_transformation(pcl1, gps_pcl, ids)


# write transformed trajectories



def get_result_dict(traj_p, traj_q, t_ns, T_gps_cam):
    out_dict = {"p_w_c": [], "q_w_c": [], "t_ns": []}
    # transform trajectories
    for i in range(len(traj_q)):
        T_gps = T_gps_cam[:3,:3] @ SE3_from_t_q(traj_p[i], traj_q[i]).matrix()
        out_dict["p_w_c"].append(T_gps[:3,3].tolist())
        out_dict["q_w_c"].append(R.from_matrix(T_gps[:3,:3]).as_quat().tolist())
        out_dict["t_ns"].append(t_ns[i])
    return out_dict

aligned_trajectories = {
    "traj1": get_result_dict(traj1_ts, traj1_qs, t_ns1, T_gps_cam),
    "traj2": get_result_dict(traj2_ts, traj2_qs, t_ns2, T_gps_cam),
    "llho": llh01.tolist()
}

a_file = open(os.path.join(base_path,"aligned_trajectories.dict"), "wb")
pickle.dump(aligned_trajectories, a_file)