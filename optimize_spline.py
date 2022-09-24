import os
import numpy as np
import pyvisfm as pvi
import pytheia as pt
from utils import load_dataset
from telemetry_converter import TelemetryImporter
import torch
from dpvo import projective_ops as pops
import cv2
from dpvo.lietorch import SE3

def reproject(ii, jj, kk, patches, intrinsics, poses):

    coords = pops.transform(poses, patches, intrinsics, ii, jj, kk)
    return coords.permute(0, 1, 4, 2, 3).contiguous()

def create_pt_recon_from_vo(vo_data, win_size):


    poses = np.concatenate(
        [np.array(dataset["p_w_c"]), np.array(dataset["q_w_c"])],1)

    start_id = 40

    # read some frames to test stuff
    ti = t_ns[start_id]
    tj = t_ns[start_id+1]

    patches = vo_data["patches"]
    pi = patches[start_id]
    num_patches_per_img = patches.shape[1]
    total_num_patches = patches.shape[0]*patches.shape[1]
    flattened_patches = torch.tensor(patches).view(1, total_num_patches, 3, 3, 3)

    
    pt_recon = pt.Reconstruction()
    for t_ in t_ns:
        
    # now project to next frames
    for idx, t in enumerate(t_ns[start_id-win_size:start_id+win_size]):
        Ij = cv2.imread(os.path.join(base_path,"run1",str(t)+".png"))

        i = torch.zeros((num_patches_per_img)) + start_id
        j = i + (idx - win_size)
        k = torch.arange(0,num_patches_per_img) + i * num_patches_per_img

        coords = reproject(
                i.long(), j.long(), k.long(), 
                flattened_patches, 
                torch.tensor(dataset["intrinsics"]).unsqueeze(0), 
                SE3(torch.tensor(np.expand_dims(poses,0))).inv())
        
        I_j_w_pts = plot_points(Ij, coords.squeeze(0))

        cv2.imshow("projected_patches",np.concatenate([I_i_w_pts, I_j_w_pts],0))
        cv2.waitKey(0)


base_path = "/media/Data/Sparsenet/Ammerbach/Links"

# load telemetry
telemetry = TelemetryImporter()
telemetry.read_gopro_telemetry(os.path.join(base_path,"bike1_trail1_linear.json"))
llh0 = telemetry.telemetry["gps_llh"][0]

# load VO dataset
dataset = load_dataset(
    os.path.join(base_path,"dpvo_result_run1.npz"),
    os.path.join(base_path,"dpvo_result_run1.json"),
    llh0, inv_depth_thresh=0.5, 
    scale_with_gps=True, align_with_grav=True, correct_heading=False)

t_ns = np.array(dataset["frametimes_ns"])

p_w_c = np.array(dataset["p_w_c"])
q_w_c = np.array(dataset["q_w_c"])

# initialize a spline
spline_estimator = pvi.SplineTrajectoryEstimator()
spline_estimator.SetT_i_c(R_i_c, np.zeros((3,1), dtype=np.float32)) #np.expand_dims(t_i_c,1))# np.expand_dims(t_i_c,1)) # np.zeros((3,1), dtype=np.float32))# np.expand_dims(t_i_c,1))
spline_estimator.SetGravity(np.array([0,0,-9.811]))
line_delay_init = 1./camera.ImageHeight()*1./telemetry.telemetry["camera_fps"]
spline_estimator.SetCameraLineDelay(line_delay_init)
spline_estimator.InitSO3R3WithVision(theia_recon, int(so3_dt*1e9), int(r3_dt*1e9))

spline_estimator.InitBiasSplines(
    np.array([0.,0.,0.]), np.array([0.,0.,0.]), int(10*1e9), int(10*1e9), 2.0, 1e-2)

print("Initial camera line delay: ", spline_estimator.GetRSLineDelay()*1e6,"us")
for i in range(len(telemetry.telemetry["gyroscope"])):
    imu_t_ns = int(imu_times[i])-int(imu_times[0])
    if imu_t_ns < start_time_ns or imu_t_ns > end_time_ns:
        continue
    spline_estimator.AddGyroscopeMeasurement(gyro_clean[:,i], imu_t_ns, gyr_weight_vec)    
    spline_estimator.AddAccelerometerMeasurement(accl_clean[:,i], imu_t_ns, accl_weight_vec)  

for vid in theia_recon.ViewIds:
    spline_estimator.AddRSCameraMeasurement(theia_recon.View(vid), theia_recon, 5.0)
    time_ns = theia_recon.View(vid).GetTimestamp() *1e9
    
# load image observations

# optimize spline

