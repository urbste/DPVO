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
from dpvo.projective_ops import iproj
from dpvo.projective_ops import point_cloud
from scipy.spatial.transform import Rotation as R
import natsort
from utils import load_camera_imu_calibration

def plot_points_debug(img, img_pts, t_id):
    for i in range(img_pts.shape[0]):
        p3 = img_pts[i]
        int_p2 = (int(p3[0]), int(p3[1]))
        img = cv2.drawMarker(img, int_p2, (0,0,255), cv2.MARKER_SQUARE, 8, 1)
        img = cv2.putText(img, t_id, int_p2, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0),1)
    return img


def get_cam_intrinsics(vo_intrinsics, scale=4):
    # create camera intrinsics
    camera = pt.sfm.Camera()
    prior = pt.sfm.CameraIntrinsicsPrior()
    intr_vo = vo_intrinsics[0,:]
    prior.aspect_ratio.value = [intr_vo[1]/intr_vo[0]]
    prior.image_width = 640
    prior.image_height = 480
    prior.principal_point.value = [intr_vo[2]*scale, intr_vo[3]*scale]
    prior.focal_length.value = [intr_vo[0]*scale]
    prior.skew.value = [0.0]
    prior.camera_intrinsics_model_type = "PINHOLE"
    camera.SetFromCameraIntrinsicsPriors(prior)
    return camera

def iproj_view(patches, intrinsics, scale=4):
    """ inverse projection """
    x, y, d = patches.unbind(dim=2)
    fx, fy, cx, cy = intrinsics * scale
    x *= scale
    y *= scale
    i = torch.ones_like(d)
    xn = (x - cx) / fx
    yn = (y - cy) / fy

    X = torch.stack([xn, yn, i, d], dim=-1)
    return X

def create_pt_recon_from_vo(vo_data, win_size):

    recon = pt.sfm.Reconstruction()

    vo_intrinsics = np.array(vo_data["intrinsics"])
    vo_intrinsics_t = torch.tensor(vo_intrinsics).float()
    pt_camera_intr = get_cam_intrinsics(vo_intrinsics, scale=4)

    poses = np.concatenate(
        [np.array(dataset["p_w_c"]), np.array(dataset["q_w_c"])],1)

    # read some frames to test stuff
    patches = np.array(vo_data["patches"])
    patches_t = torch.tensor(patches).float()
    # num_patches_per_img = patches.shape[1]
    #total_num_patches = patches.shape[0]*patches.shape[1]
    # flattened_patches = torch.tensor(patches).view(1, total_num_patches, 3, 3, 3)

    # first add patches for each view
    view_ids_to_pt_ids = {}
    for view_id in range(patches.shape[0]):
        # first create points from patches
        X0 = iproj_view(patches_t[[view_id],...], vo_intrinsics_t[view_id,:], 4)
        view_ids_to_pt_ids[view_id] = []
        for i in range(X0.shape[1]):
            pose_w_c_t = SE3(torch.tensor(poses[view_id,:]).float())
            pt_in_cam = pose_w_c_t.matrix().numpy() @ X0[0, i, 1, 1, :].numpy()
            pt_in_cam = pt_in_cam/pt_in_cam[3]
            t_id = recon.AddTrack()
            point = recon.MutableTrack(t_id)
            point.SetPoint(pt_in_cam)
            point.SetIsEstimated(True)
            view_ids_to_pt_ids[view_id].append(t_id)

    # now add views and observations
    t_ns = vo_data["frametimes_ns"]
    for i, t_ in enumerate(t_ns):
        v_id = recon.AddView(str(t_), 0, t_*1e-9)
        v = recon.MutableView(v_id)
        R_w_c = R.from_quat(poses[i,3:]).as_matrix()
        cam = v.MutableCamera()
        cam.DeepCopy(pt_camera_intr)
        cam.SetPosition(poses[i,0:3])
        cam.SetOrientationFromRotationMatrix(R_w_c.T)
        v.SetIsEstimated(True)

    # now create observations by projecting to neighboring frames
    v_ids = natsort.natsorted(recon.ViewIds)
    start_id = v_ids[win_size]
    end_id = v_ids[-win_size]
    for v_id in range(start_id,end_id):
        v_ids_neighbor = list(range(v_id-win_size, v_id + win_size))
        # get points ids in that view
        track_ids = view_ids_to_pt_ids[v_id]
        for t_id in track_ids:
            
            for v_n in v_ids_neighbor:
                d, pt2 = recon.View(v_n).Camera().ProjectPoint(recon.Track(t_id).Point())
                # check if point is in image and depth is not too large or too small
                if d > 20 or d < 0.1:
                    continue
                if pt2[0] < 0.0 or pt2[1] > pt_camera_intr.ImageWidth() or \
                   pt2[1] < 0.0 or pt2[1] > pt_camera_intr.ImageHeight():
                   continue
                recon.AddObservation(v_n, t_id, pt.sfm.Feature(pt2, np.eye(2)*2))
    # do the same for the start and end frames
    for v_id in range(0, win_size-1):
        v_ids_neighbor = list(range(v_id, v_id+win_size))
        track_ids = view_ids_to_pt_ids[v_id]
        for t_id in track_ids:
            for v_n in v_ids_neighbor:
                d, pt2 = recon.View(v_n).Camera().ProjectPoint(recon.Track(t_id).Point())
                # check if point is in image and depth is not too large or too small
                if d > 20 or d < 0.1:
                    continue
                if pt2[0] < 0.0 or pt2[1] > pt_camera_intr.ImageWidth() or \
                   pt2[1] < 0.0 or pt2[1] > pt_camera_intr.ImageHeight():
                   continue
                recon.AddObservation(v_n, t_id, pt.sfm.Feature(pt2, np.eye(2)*2))
    # do the same for the start and end frames
    for v_id in range(v_ids[-win_size+1], v_ids[-1]+1):
        v_ids_neighbor = list(range(v_id - win_size + 1, v_id + 1))
        track_ids = view_ids_to_pt_ids[v_id]
        for t_id in track_ids:
            for v_n in v_ids_neighbor:
                d, pt2 = recon.View(v_n).Camera().ProjectPoint(recon.Track(t_id).Point())
                # check if point is in image and depth is not too large or too small
                if d > 20 or d < 0.1:
                    continue
                if pt2[0] < 0.0 or pt2[1] > pt_camera_intr.ImageWidth() or \
                   pt2[1] < 0.0 or pt2[1] > pt_camera_intr.ImageHeight():
                   continue
                recon.AddObservation(v_n, t_id, pt.sfm.Feature(pt2, np.eye(2)*2))

    print("Reconstruction created. Statistics:")
    # print("Number views {}. Number tracks: {}")

    # calculate reference bearing vectors
    for tid in recon.TrackIds:
        mut_track = recon.MutableTrack(tid)
        if mut_track.ReferenceViewId() == pt.sfm.kInvalidViewId:
            recon.RemoveTrack(tid)
            continue
        ref_view = recon.View(mut_track.ReferenceViewId())
        feature = ref_view.GetFeature(tid)
        ref_cam = ref_view.Camera()
        bearing = ref_cam.PixelToNormalizedCoordinates(feature.point)
        d, _ = ref_cam.ProjectPoint(mut_track.Point())
        mut_track.SetReferenceBearingVector(bearing)
        mut_track.SetInverseDepth(1/d)

    # I1 = cv2.imread(os.path.join(base_path,"run1",str(t_ns[3])+".png"))
    # I2 = cv2.imread(os.path.join(base_path,"run1",str(t_ns[4])+".png"))
    # I3 = cv2.imread(os.path.join(base_path,"run1",str(t_ns[5])+".png"))
    # Is = [I1, I2, I3]
    # # get track_ids of central view
    # t_ids_in_view = view_ids_to_pt_ids[4] # recon.View(4).TrackIds()
    # for t_id in t_ids_in_view:
    #     for v_id in range(3,6):
    #         feature = recon.View(v_id).GetFeature(t_id)
    #         if feature:
    #             pt2 = feature.point
    #             int_p2 = (int(pt2[0]), int(pt2[1]))
    #             Is[v_id-3] = cv2.drawMarker(Is[v_id-3], int_p2, (0,0,255), cv2.MARKER_SQUARE, 8, 1)
    #             Is[v_id-3] = cv2.putText(Is[v_id-3], str(t_id), int_p2, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0),1)

    # #I = plot_points_debug(I, np.array(pts_debug), [t_id])
    # cv2.imshow("projected_patches", np.concatenate(Is,1))
    # cv2.waitKey(0)

    return recon, pt_camera_intr

base_path = "/media/Data/Sparsenet/Ammerbach/Links"

# load telemetry
telemetry = TelemetryImporter()
telemetry.read_gopro_telemetry(os.path.join(base_path,"run1.json"))
llh0 = telemetry.telemetry["gps_llh"][0]

# load VO dataset
dataset = load_dataset(
    os.path.join(base_path,"dpvo_result_run1.npz"),
    os.path.join(base_path,"run1.json"),
    llh0, inv_depth_thresh=0.5, 
    scale_with_gps=True, align_with_grav=True, correct_heading=False)

q_r3 = 0.97
q_so3 = 0.97
acc_white_noise_var = 0.0196602**2
gyr_white_noise_var = 0.00154431**2
gyro_np = np.array(dataset["gyro"])
accl_np = np.array(dataset["accl"])
imu_times_ns = np.array(dataset["imu_times"])

from sew import knot_spacing_and_variance
r3_dt, r3_var,_,_ = knot_spacing_and_variance(
    accl_np.T, imu_times_ns*1e-9, q_r3, min_dt=0.05, 
    max_dt=.5, verbose=False, measurement_variance=acc_white_noise_var)
so3_dt, so3_var,_,_ = knot_spacing_and_variance(
    gyro_np.T, imu_times_ns*1e-9, q_so3, min_dt=0.05, 
    max_dt=.5, verbose=False, measurement_variance=gyr_white_noise_var)

#gyr_weight_vec = [1/np.sqrt(g_wx+gyr_var_n),1/np.sqrt(g_wy+gyr_var_n),1/np.sqrt(g_wz+gyr_var_n)]
#accl_weight_vec = [1/np.sqrt(a_wx+acc_var_n),1/np.sqrt(a_wy+acc_var_n),1/np.sqrt(a_wz+acc_var_n)]
gyr_weight_vec = [1/np.sqrt(so3_var),1/np.sqrt(so3_var),1/np.sqrt(so3_var)]
accl_weight_vec = [1/np.sqrt(r3_var),1/np.sqrt(r3_var),1/np.sqrt(r3_var)] # weight z down, there is no movement in that direction
print("Gyroscope weighting factor:     {:.3f}/{:.3f}/{:.3f}. Knot spacing SO3: {:.3f}".format(
    gyr_weight_vec[0],gyr_weight_vec[1],gyr_weight_vec[2],so3_dt))
print("Accelerometer weighting factor: {:.3f}/{:.3f}/{:.3f}. Knot spacing R3: {:.3f}".format(
    accl_weight_vec[0],accl_weight_vec[1],accl_weight_vec[2],r3_dt))

print("Creating a pyTheiaSfM reconstruction.")
recon, camera = create_pt_recon_from_vo(dataset, 3)
print("Finished creating a pyTheiaSfM reconstruction.")
R_i_c, t_i_c, T_i_c = load_camera_imu_calibration("calib/cam_imu_calib_result_GX018770.json")
# initialize a spline
spline_estimator = pvi.SplineTrajectoryEstimator()
spline_estimator.SetT_i_c(R_i_c, np.expand_dims(t_i_c,1)) #np.expand_dims(t_i_c,1))# np.expand_dims(t_i_c,1)) # np.zeros((3,1), dtype=np.float32))# np.expand_dims(t_i_c,1))
spline_estimator.SetGravity(np.array([0,0,9.811]))
line_delay_init = 1./camera.ImageHeight()*1./telemetry.telemetry["camera_fps"]
spline_estimator.SetCameraLineDelay(line_delay_init)
spline_estimator.InitSO3R3WithVision(recon, int(so3_dt*1e9), int(r3_dt*1e9))
spline_estimator.InitBiasSplines(
    np.array([0.,0.,0.]), np.array([0.,0.,0.]), int(10*1e9), int(10*1e9), 1.0, 1e-2)

print("Initial camera line delay: ", spline_estimator.GetRSLineDelay()*1e6,"us")
start_time_ns = dataset["frametimes_ns"][0]
end_time_ns = dataset["frametimes_ns"][-1]
accl_meas = {}
for i in range(len(telemetry.telemetry["gyroscope"])):
    imu_t_ns = int(imu_times_ns[i])-int(imu_times_ns[0])
    if imu_t_ns < start_time_ns or imu_t_ns > end_time_ns:
        continue
    spline_estimator.AddGyroscopeMeasurement(gyro_np[i,:], imu_t_ns, gyr_weight_vec)    
    spline_estimator.AddAccelerometerMeasurement(accl_np[i,:], imu_t_ns, accl_weight_vec)  
    accl_meas[imu_t_ns] = accl_np[i,:].tolist()
for vid in recon.ViewIds:
    spline_estimator.AddRSCameraMeasurement(recon.View(vid), recon, 4.0)
    

# optimize spline
flags = pvi.SplineOptimFlags.SO3 | pvi.SplineOptimFlags.R3 | pvi.SplineOptimFlags.POINTS |  pvi.SplineOptimFlags.IMU_BIASES 
print("Init line delay: ",spline_estimator.GetRSLineDelay())
spline_estimator.OptimizeFromTo(10, flags, recon, 0, 0)

# accl_tns = sorted(list(accl_meas.keys()))
# pose = spline_estimator.GetPose(accl_tns[10])
# R_w_i = R.from_quat(pose[0:4]).as_matrix()
# accl_world = spline_estimator.GetAccelerationWorld(accl_tns[10])
# test = R_w_i.T @ np.array([0,0,9.81]) + 

optim_gyro_bias = spline_estimator.GetGyroBias(int(5*1e9))
optim_accl_bias = spline_estimator.GetAcclBias(int(5*1e9))
print("Optimized gyro bias: ", optim_gyro_bias)
print("Optimized acclbias: ", optim_accl_bias)

# camera_poses = {}
# camera_c_w = {}
# for v_id in recon.ViewIds:
#     view = recon.View(v_id)
#     t_ns = view.GetTimestamp()*1e9 
#     camera_c_w[t_ns] = recon.View(v_id).Camera().GetOrientationAsAngleAxis()
#     camera_poses[t_ns] = recon.View(v_id).Camera().GetPosition()


pt.io.WritePlyFile(os.path.join(base_path,"recon_output.ply"), recon, (255, 0, 0), 2)
spline_estimator.UpdateCameraPoses(recon, False)
pvi.WriteSpline(spline_estimator, os.path.join(base_path,"spline_recon_run1.spline"))
pt.io.WriteReconstruction(recon, os.path.join(base_path,"spline_recon_run1.recon"))
pt.io.WritePlyFile(os.path.join(base_path,"spline_output_run1.ply"), recon, (255, 0, 0), 2)