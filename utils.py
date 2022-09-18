import numpy as np
from telemetry_converter import TelemetryImporter
from natsort import natsorted
from gps_converter import ECEFtoENU
from dpvo.lietorch import SE3
import torch

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
    
    return p_w_c.numpy(), q_w_c.numpy(), gravity_vectors, np.array(gps_enu_at_kfs, dtype=np.float32), frametimes_slam_ns