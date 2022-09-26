import numpy as np
from telemetry_converter import TelemetryImporter
from natsort import natsorted
from gps_converter import ECEFtoNED
from dpvo.lietorch import SE3
import torch

from scipy.spatial.transform import Rotation as R
from trafo_utils import get_vis_scaler, get_rot_to_worldframe
from trafo_utils import get_heading_angle_diff
import json
from pymap3d import ecef2ned

def ecef2ned_v(ecef, ll0):
    return ecef2ned(ecef[0],ecef[1],ecef[2], ll0[0], ll0[1], ll0[2])

def load_dataset(path, telemetry_file, llh0, inv_depth_thresh=0.2, 
    scale_with_gps=False, align_with_grav=True, correct_heading=False):

    data = np.load(path)
    poses_w_c = data["kf_poses"]
    num_kfs = poses_w_c.shape[0]
    frametimes_slam_ns = data["image_tstamps"].astype(np.int64)[:num_kfs]
    frame_ids = data["tstamps"][:num_kfs]
    patches = data["patches"][:num_kfs,...]
    ii, jj, kk, ix = data["ii"], data["jj"], data["kk"], data["ix"]
    
    large_depths = patches[:,:,2] > inv_depth_thresh
    pt_mask = large_depths[:,:,0,0].reshape(-1)
    valid_points = data["points"][pt_mask]
    valid_point_colors = data["pt_colors"].reshape(-1,3)[pt_mask] / 255.

    tel_importer = TelemetryImporter()
    tel_importer.read_gopro_telemetry(telemetry_file)
    gps_xyz, _ = tel_importer.get_gps_pos_at_frametimes(frametimes_slam_ns)
    gravity_vectors = tel_importer.get_gravity_vector_at_times(frametimes_slam_ns)
    if llh0 == None:
        llh0 = tel_importer.telemetry["gps_llh"][0]
    
    gps_ned_at_kfs = np.asarray([
        ecef2ned_v(gps_xyz[int(key)], llh0) \
            if int(key) in gps_xyz else print(key) for key in frametimes_slam_ns])

    p_w_c = SE3(torch.tensor(poses_w_c)).translation()[:,0:3].numpy()
    q_w_c = SE3(torch.tensor(poses_w_c)).data[:,3:].numpy()

    s = 1
    if scale_with_gps:
        s = get_vis_scaler(p_w_c, gps_ned_at_kfs)
        p_w_c = s * p_w_c
        valid_points = s * valid_points
        patches[:,:,2] /= s
        
    # gravity normalization and scale
    R_to_grav = np.eye(3)
    if align_with_grav:
        R_to_grav = get_rot_to_worldframe(gravity_vectors, q_w_c, world_vec=np.array([0,0,1.]))
        p_w_c = (R_to_grav @ p_w_c.T).T
        q_w_c = R.from_matrix(R.from_quat(q_w_c).inv().as_matrix() @ R_to_grav.T).inv().as_quat()
        valid_points = (R_to_grav @ valid_points.T).T

    gps_normalized = gps_ned_at_kfs-gps_ned_at_kfs[0]
    R_heading = np.eye(3)
    if correct_heading:
        R_heading = R.from_rotvec([0,0,-get_heading_angle_diff(p_w_c, gps_normalized)]).as_matrix()
        p_w_c = (R_heading @ p_w_c.T).T
        q_w_c = R.from_matrix(R.from_quat(q_w_c).inv().as_matrix() @ R_heading.T).inv().as_quat()
        valid_points = (R_heading @ valid_points.T).T

    dataset = {
        "patches": patches.tolist(),
        "points": valid_points.tolist(),
        "colors": valid_point_colors.tolist(),
        "intrinsics": data["intrinsics"].tolist(),
        "p_w_c": p_w_c.tolist(),
        "q_w_c": q_w_c.tolist(),
        "R_to_grav": R_to_grav.tolist(),
        "R_heading": R_heading.tolist(),
        "map_scale": s,
        "gravity_vectors": gravity_vectors.tolist(),
        "gps_local_ned": gps_ned_at_kfs.tolist(),
        "frametimes_ns": frametimes_slam_ns.tolist(),
        "frame_ids": frame_ids,
        "ix": ix, "ii": ii, "kk": kk, "jj": jj,
        "accl": tel_importer.telemetry["accelerometer"],
        "gyro": tel_importer.telemetry["gyroscope"],
        "imu_times": tel_importer.telemetry["timestamps_ns"]
    }

    return dataset

def load_pytheia_cam_calib(datapath, scale=1):
    import pytheia as pt

    with open(datapath, 'r') as f:
        camera_intrinsics = json.load(f)

    camera = pt.sfm.Camera()

    prior = pt.sfm.CameraIntrinsicsPrior()
    prior.aspect_ratio.value = [camera_intrinsics["intrinsics"]["aspect_ratio"]]
    prior.image_width = int(camera_intrinsics["image_width"]*scale)
    prior.image_height = int(camera_intrinsics["image_height"]*scale)
    prior.principal_point.value = [camera_intrinsics["intrinsics"]["principal_pt_x"]*scale, 
                                   camera_intrinsics["intrinsics"]["principal_pt_y"]*scale]
    prior.focal_length.value = [camera_intrinsics["intrinsics"]["focal_length"]*scale]
    prior.skew.value = [camera_intrinsics["intrinsics"]["skew"]]
    
    if camera_intrinsics["intrinsic_type"] == "PINHOLE_RADIAL_TANGENTIAL":
        prior.radial_distortion.value = [camera_intrinsics["intrinsics"]["radial_distortion_1"],
            camera_intrinsics["intrinsics"]["radial_distortion_2"], camera_intrinsics["intrinsics"]["radial_distortion_3"], 0]
        prior.tangential_distortion.value = [camera_intrinsics["intrinsics"]["tangential_distortion_1"],
            camera_intrinsics["intrinsics"]["tangential_distortion_2"]]
    prior.camera_intrinsics_model_type = camera_intrinsics["intrinsic_type"] 
    camera.SetFromCameraIntrinsicsPriors(prior)
    if camera_intrinsics["intrinsic_type"] != "ORTHOGRAPHIC":
        dist = [camera_intrinsics["intrinsics"]["radial_distortion_1"],     
                camera_intrinsics["intrinsics"]["radial_distortion_2"],
                camera_intrinsics["intrinsics"]["tangential_distortion_1"],
                camera_intrinsics["intrinsics"]["tangential_distortion_2"],
                camera_intrinsics["intrinsics"]["radial_distortion_3"]]
    else: dist = []
    return camera, prior.image_width, prior.image_height, dist

def load_camera_imu_calibration(user_calib):

    with open(user_calib, 'r') as f:
        imu_camera = json.load(f)
    R_imu_cam = R.from_quat([imu_camera["q_i_c"]["x"],imu_camera["q_i_c"]["y"],imu_camera["q_i_c"]["z"], imu_camera["q_i_c"]["w"]])
    #R_cam_imu = R_imu_cam.inv().as_matrix()
    t_imu_cam = np.array([imu_camera["t_i_c"]["x"],imu_camera["t_i_c"]["y"],imu_camera["t_i_c"]["z"]]).T
    # t_cam_imu = -R_imu_cam.as_matrix().T @ t_imu_cam
    T_imu_cam = np.eye(4,dtype=np.float32)
    T_imu_cam[:3,:3] = R_imu_cam.as_matrix()
    T_imu_cam[:3,3] = t_imu_cam

    return R_imu_cam.as_matrix(), t_imu_cam, T_imu_cam