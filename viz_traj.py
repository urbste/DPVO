import open3d as o3d
import numpy as np
import os
from telemetry_converter import TelemetryImporter

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

dataset1 = load_dataset(
    os.path.join(base_path,"dpvo_result_bike1_trail1_linear.npz"),
    os.path.join(base_path,"bike1_trail1_linear.json"),
    llh01, inv_depth_thresh=0.5, scale_with_gps=True, align_with_grav=True, correct_heading=True)

dataset2 = load_dataset(
    os.path.join(base_path,"dpvo_result_bike2_trail1_linear.npz"),
    os.path.join(base_path,"bike2_trail1_linear.json"),
    llh01, inv_depth_thresh=0.5, scale_with_gps=True, align_with_grav=True, correct_heading=True)


pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(dataset1["p_w_c"])
pcl1.paint_uniform_color([0, 0.706, 0])

pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector(dataset1["p_w_c"]-np.array([0,0,1.5]))
pcl2.paint_uniform_color([1, 0, 0])

#pcl2 = o3d.geometry.PointCloud()
#pcl2.points = o3d.utility.Vector3dVector(dataset2["p_w_c"])
#pcl2.paint_uniform_color([1, 1,  0.706])

map_pts_pcl1 = o3d.geometry.PointCloud()
map_pts_pcl1.points = o3d.utility.Vector3dVector(dataset1["points"])
map_pts_pcl1.colors = o3d.utility.Vector3dVector(dataset1["colors"])
#map_pts_pcl2 = o3d.geometry.PointCloud()
#map_pts_pcl2.points = o3d.utility.Vector3dVector(dataset2["points"])
#map_pts_pcl2.colors = o3d.utility.Vector3dVector(dataset2["colors"])

# map_pts_pcl1.estimate_normals()
# map_pts_pcl2.estimate_normals()

# o3d.io.write_point_cloud(os.path.join(base_path,"pcl1.ply"), map_pts_pcl1)

# radii = [1]
# mesh1 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     map_pts_pcl1, o3d.utility.DoubleVector(radii))
# mesh1.compute_vertex_normals()

gps_pcl1 = o3d.geometry.PointCloud()
gps_np = np.array(dataset1["gps_local_ned"])
gps_pcl1.points = o3d.utility.Vector3dVector(gps_np-gps_np[0,:])
gps_pcl1.paint_uniform_color([1, 0.5, 0.5])
# gps_pcl2 = o3d.geometry.PointCloud()
# gps_pcl2.points = o3d.utility.Vector3dVector(gps2-gps2[0])
# gps_pcl2.paint_uniform_color([1, 0,  0.706])


world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=600, height=500, left=450, top=250)
visualizer.add_geometry(pcl1)
visualizer.add_geometry(pcl2)
visualizer.add_geometry(map_pts_pcl1)
p_w_cs = np.array(dataset1["p_w_c"])
q_w_cs = np.array(dataset1["q_w_c"])

from scipy.spatial.transform import Rotation as R 
for i in range(p_w_cs.shape[0]):
    T = np.eye(4)
    T[:3,:3] = R.from_quat(q_w_cs[i,:]).as_matrix()
    T[:3,3] = p_w_cs[i,:]
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    visualizer.add_geometry(cam_frame.transform(T))

#visualizer.add_geometry(mesh1)
#visualizer.add_geometry(map_pts_pcl2)
visualizer.add_geometry(gps_pcl1)
#visualizer.add_geometry(gps_pcl2)
visualizer.add_geometry(world_frame)

view_ctl = visualizer.get_view_control()
view_ctl.set_front((1, 0, 0))
view_ctl.set_up((0, 0, 1))  # can't set z-axis as the up-axis
view_ctl.set_lookat((0, 0, 0))

visualizer.run()
visualizer.destroy_window()


