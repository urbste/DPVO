import numpy as np
from telemetry_converter import TelemetryImporter
from natsort import natsorted
from gps_converter import ECEFtoENU
from dpvo.lietorch import SE3
import torch

from scipy.spatial.transform import Rotation as R
from trafo_utils import get_vis_scaler, get_rot_to_worldframe
from trafo_utils import get_heading_angle_diff


def load_dataset(path, telemetry_file, llh0, inv_depth_thresh=0.2, 
    scale_with_gps=False, align_with_grav=True, correct_heading=False):

    data = np.load(path)
    poses_w_c = data["kf_poses"]
    num_kfs = poses_w_c.shape[0]
    frametimes_slam_ns = data["image_tstamps"].astype(np.int64)[:num_kfs]
    frame_ids = data["tstamps"][:num_kfs]
    patches = data["patches"][:num_kfs,...]
    ii, jj, kk = data["ii"], data["jj"], data["kk"]
    
    large_depths = patches[:,:,2] > inv_depth_thresh
    pt_mask = large_depths[:,:,0,0].reshape(-1)
    valid_points = data["points"][pt_mask]
    valid_point_colors = data["pt_colors"].reshape(-1,3)[pt_mask] / 255.

    tel_importer = TelemetryImporter()
    tel_importer.read_gopro_telemetry(telemetry_file)
    gps_xyz, _ = tel_importer.get_gps_pos_at_frametimes(frametimes_slam_ns)
    gravity_vectors = tel_importer.get_gravity_vector_at_times(frametimes_slam_ns)

    p_w_c = SE3(torch.tensor(poses_w_c)).translation()[:,0:3].numpy()
    q_w_c = SE3(torch.tensor(poses_w_c)).data[:,3:].numpy()

    gps_enu_at_kfs = np.asarray([ECEFtoENU(gps_xyz[int(key)], llh0) if int(key) in gps_xyz else print(key) for key in frametimes_slam_ns])

    s = 1
    if scale_with_gps:
        s = get_vis_scaler(p_w_c, gps_enu_at_kfs)
        p_w_c = s * p_w_c
        valid_points = s * valid_points
    
    # gravity normalization and scale
    R_to_grav = np.eye(3)
    if align_with_grav:
        R_to_grav = get_rot_to_worldframe(gravity_vectors, q_w_c)
        p_w_c = (R_to_grav @ p_w_c.T).T
        q_w_c = R.from_matrix(R.from_quat(q_w_c).inv().as_matrix() @ R_to_grav.T).inv().as_quat()
        valid_points = (R_to_grav @ valid_points.T).T

    gps_normalized = gps_enu_at_kfs-gps_enu_at_kfs[0]
    R_heading = np.eye(3)
    if correct_heading:
        R_heading = R.from_rotvec([0,0,-get_heading_angle_diff(p_w_c, gps_normalized)]).as_matrix()
        p_w_c = (R_heading @ p_w_c.T).T
        q_w_c = R.from_matrix(R.from_quat(q_w_c).inv().as_matrix() @ R_to_grav.T).inv().as_quat()
        valid_points = (R_heading @ valid_points.T).T

    dataset = {
        "points": valid_points.tolist(),
        "colors": valid_point_colors.tolist(),
        "p_w_c": p_w_c.tolist(),
        "q_w_c": q_w_c.tolist(),
        "R_to_grav": R_to_grav.tolist(),
        "R_heading": R_heading.tolist(),
        "map_scale": s,
        "gravity_vectors": gravity_vectors.tolist(),
        "gps_enu": gps_enu_at_kfs.tolist(),
        "frametimes_ns": frametimes_slam_ns.tolist()
    }

    return dataset