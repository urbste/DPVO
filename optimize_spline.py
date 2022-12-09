import os
import numpy as np
import pyvisfm as pvi
import pytheia as pt
from utils import load_dataset, update_dataset_patches
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
        [np.array(vo_data["p_w_c"]), np.array(vo_data["q_w_c"])],1)

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
    t_ns = sorted(vo_data["frametimes_slam_ns"]) # starts at 0 with video time
    for i, t_ in enumerate(t_ns):
        v_id = recon.AddView(str(t_), 0, t_*1e-9)
        v = recon.MutableView(v_id)
        R_w_c = R.from_quat(poses[i,3:]).as_matrix()
        cam = v.MutableCamera()
        cam.DeepCopy(pt_camera_intr)
        cam.SetPosition(poses[i,0:3])
        cam.SetOrientationFromRotationMatrix(R_w_c.T)
        v.SetIsEstimated(True)
        
    std_2d = 1.0
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
                if d > 10 or d < 0.1:
                    continue
                if pt2[0] < 0.0 or pt2[1] > pt_camera_intr.ImageWidth() or \
                   pt2[1] < 0.0 or pt2[1] > pt_camera_intr.ImageHeight():
                   continue
                recon.AddObservation(v_n, t_id, pt.sfm.Feature(pt2, np.eye(2)*std_2d))
    
    # do the same for the start and end frames
    for v_id in range(0, win_size-1):
        v_ids_neighbor = list(range(v_id, v_id+win_size))
        track_ids = view_ids_to_pt_ids[v_id]
        for t_id in track_ids:
            for v_n in v_ids_neighbor:
                d, pt2 = recon.View(v_n).Camera().ProjectPoint(recon.Track(t_id).Point())
                # check if point is in image and depth is not too large or too small
                if d > 10 or d < 0.1:
                    continue
                if pt2[0] < 0.0 or pt2[1] > pt_camera_intr.ImageWidth() or \
                   pt2[1] < 0.0 or pt2[1] > pt_camera_intr.ImageHeight():
                   continue
                recon.AddObservation(v_n, t_id, pt.sfm.Feature(pt2, np.eye(2)*std_2d))
    # do the same for the start and end frames
    for v_id in range(v_ids[-win_size+1], v_ids[-1]+1):
        v_ids_neighbor = list(range(v_id - win_size + 1, v_id + 1))
        track_ids = view_ids_to_pt_ids[v_id]
        for t_id in track_ids:
            for v_n in v_ids_neighbor:
                d, pt2 = recon.View(v_n).Camera().ProjectPoint(recon.Track(t_id).Point())
                # check if point is in image and depth is not too large or too small
                if d > 10 or d < 0.1:
                    continue
                if pt2[0] < 0.0 or pt2[1] > pt_camera_intr.ImageWidth() or \
                   pt2[1] < 0.0 or pt2[1] > pt_camera_intr.ImageHeight():
                   continue
                recon.AddObservation(v_n, t_id, pt.sfm.Feature(pt2, np.eye(2)*std_2d))

    print("Reconstruction created. Statistics:")
    # print("Number views {}. Number tracks: {}")

    # calculate reference bearing vectors
    for tid in recon.TrackIds:
        mut_track = recon.MutableTrack(tid)
        if mut_track.ReferenceViewId() == pt.sfm.kInvalidViewId:
           recon.RemoveTrack(tid)
        #    print("here why?")
        #    print(tid)
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

    return recon, pt_camera_intr, view_ids_to_pt_ids

base_path = "/media/Data/Sparsenet/Ammerbach/Links"
run = "run1"
tel_json = run+".json"
dpvo_res = "dpvo_result_"+run+".npz"

add_base_trail_constrains = False
# time trail2 -> trail1 : position constrain, rotation constrain
# const_to_base_trail = {
#     2316785000: {"time_base": 3136812000},
#     5616785000: {"time_base": 6516812000},
#     6816785000: {"time_base": 7816812000},
#     10696785000: {"time_base": 11816812000},
#     13116785000: {"time_base": 14176812000},
#     17716785000: {"time_base": 19016812000},
#     21696785000: {"time_base": 23256812000},
#     23676785000: {"time_base": 25176812000},
#     26416785000: {"time_base": 27936812000},
#     40516785000: {"time_base": 42516812000},
#     103816785000: {"time_base": 106496812000},
#     137716785000: {"time_base": 145816812000}
# }


# load telemetry
telemetry = TelemetryImporter()
telemetry.read_gopro_telemetry(os.path.join(base_path,tel_json))

# load VO dataset
dataset = load_dataset(
    os.path.join(base_path,dpvo_res),
    os.path.join(base_path,tel_json),
    None, inv_depth_thresh=0.5, 
    scale_with_gps=True, align_with_grav=True, correct_heading=True)

constrains = {}
if add_base_trail_constrains:
    import json
    with open(os.path.join(base_path, "matching_kfs_run1_run2.json"), "r") as f:
        ref_to_query_constrain = json.load(f)
    base_trail_spline_path = os.path.join(base_path, "spline_recon_run1.spline")
    print("Loading base trail spline from {}.".format(base_trail_spline_path))

    base_spline = pvi.SplineTrajectoryEstimator()
    pvi.ReadSpline(base_spline, base_trail_spline_path)

    # get constrains
    print("Adding {} position constrains.".format(len(ref_to_query_constrain)))
    for ref_t_ns in ref_to_query_constrain:
        query_t_ns = ref_to_query_constrain[ref_t_ns]
        base_p_w_i = base_spline.GetPose(int(ref_t_ns))
        constrains[query_t_ns] = base_p_w_i


q_r3 = 0.9999
q_so3 = 0.98
acc_white_noise_var = 0.0196602**2
gyr_white_noise_var = 0.00154431**2
gyro_np = np.array(dataset["gyro"])
accl_np = np.array(dataset["accl"])
imu_times_ns = np.array(dataset["imu_times_ns"])

from sew import knot_spacing_and_variance
r3_dt, r3_var,_,_ = knot_spacing_and_variance(
    accl_np.T, imu_times_ns*1e-9, q_r3, min_dt=1/telemetry.telemetry["camera_fps"]*2, 
    max_dt=.5, verbose=False, measurement_variance=acc_white_noise_var)
so3_dt, so3_var,_,_ = knot_spacing_and_variance(
    gyro_np.T, imu_times_ns*1e-9, q_so3, min_dt=1/telemetry.telemetry["camera_fps"]*2, 
    max_dt=.5, verbose=False, measurement_variance=gyr_white_noise_var)

#gyr_weight_vec = [1/np.sqrt(g_wx+gyr_var_n),1/np.sqrt(g_wy+gyr_var_n),1/np.sqrt(g_wz+gyr_var_n)]
#accl_weight_vec = [1/np.sqrt(a_wx+acc_var_n),1/np.sqrt(a_wy+acc_var_n),1/np.sqrt(a_wz+acc_var_n)]
gyr_weight_vec = [1/np.sqrt(so3_var),1/np.sqrt(so3_var),1/np.sqrt(so3_var)]
accl_weight_vec = [1/np.sqrt(r3_var)*4,1/np.sqrt(r3_var)*4,1/np.sqrt(r3_var)*4] # weight z down, there is no movement in that direction
print("Gyroscope weighting factor:     {:.3f}/{:.3f}/{:.3f}. Knot spacing SO3: {:.3f}".format(
    gyr_weight_vec[0],gyr_weight_vec[1],gyr_weight_vec[2],so3_dt))
print("Accelerometer weighting factor: {:.3f}/{:.3f}/{:.3f}. Knot spacing R3: {:.3f}".format(
    accl_weight_vec[0],accl_weight_vec[1],accl_weight_vec[2],r3_dt))

print("Creating a pyTheiaSfM reconstruction.")
recon, camera, view_ids_to_pt_ids = create_pt_recon_from_vo(dataset, 2)
print("Finished creating a pyTheiaSfM reconstruction.")
R_i_c, t_i_c, T_i_c = load_camera_imu_calibration("calib/cam_imu_calib_result_GX018770.json")
# initialize a spline
spline_estimator = pvi.SplineTrajectoryEstimator()
spline_estimator.SetT_i_c(R_i_c, np.expand_dims(t_i_c,1)) #np.expand_dims(t_i_c,1))# np.expand_dims(t_i_c,1)) # np.zeros((3,1), dtype=np.float32))# np.expand_dims(t_i_c,1))
spline_estimator.SetGravity(np.array([0,0,-9.811]))
line_delay_init = 1./camera.ImageHeight()*1./telemetry.telemetry["camera_fps"]
spline_estimator.SetCameraLineDelay(line_delay_init)
spline_estimator.InitSO3R3WithVision(recon, int(so3_dt*1e9), int(r3_dt*1e9))
spline_estimator.InitBiasSplines(
    np.array([0.,0.,0.]), np.array([0.,0.,0.]), int(10*1e9), int(10*1e9), 2.0, 1e-2)

print("Initial camera line delay: ", spline_estimator.GetRSLineDelay()*1e6,"us")
start_time_ns = dataset["frametimes_slam_ns"][0]
end_time_ns = dataset["frametimes_slam_ns"][-1]
accl_meas, gyro_meas = {}, {}
for i in range(len(telemetry.telemetry["gyroscope"])):
    imu_t_ns = int(imu_times_ns[i])
    if imu_t_ns < start_time_ns or imu_t_ns > end_time_ns:
        continue
    spline_estimator.AddGyroscopeMeasurement(gyro_np[i,:], imu_t_ns, gyr_weight_vec)    
    spline_estimator.AddAccelerometerMeasurement(accl_np[i,:], imu_t_ns, accl_weight_vec)  
    accl_meas[imu_t_ns] = accl_np[i,:].tolist()
    gyro_meas[imu_t_ns] = gyro_np[i,:].tolist()

accl_tns = sorted(list(accl_meas.keys()))

for vid in recon.ViewIds:
    spline_estimator.AddRSInvCameraMeasurement(recon.View(vid), recon, 6.0)

# add pose graph constrains from base trail
if add_base_trail_constrains:
    print("Adding base trail constrains")
    std_position_constrain = 0.01
    std_orientation_constrain = 10 * np.pi/180
    for t_ns in constrains:
        spline_estimator.AddPositionMeasurement(
            constrains[t_ns][4:],t_ns,1/std_position_constrain)
        spline_estimator.AddOrientationMeasurement(
            R.from_quat(constrains[t_ns][:4]).as_matrix(),
            t_ns, 1/std_orientation_constrain)

# add some GPS measurements
if not add_base_trail_constrains:
    print("Adding GPS constrains")
    gps_ned_t = np.array(dataset["gps_local_ned"])
    std_gps = 5.
    gps_weight = np.array([1/std_gps, 1/std_gps, 1/(2*std_gps)])
    for idx, t_ns in enumerate(dataset["frametimes_slam_ns"]):
        spline_estimator.AddGPSMeasurement(gps_ned_t[idx,:]-gps_ned_t[0,:], t_ns, gps_weight)

# optimize spline
flags = pvi.SplineOptimFlags.SO3 | pvi.SplineOptimFlags.R3 | pvi.SplineOptimFlags.POINTS  | pvi.SplineOptimFlags.IMU_BIASES
print("Init line delay: ",spline_estimator.GetRSLineDelay())
spline_estimator.OptimizeFromTo(15, flags, recon, 0, 0)

# pt.io.WritePlyFile(os.path.join(base_path,"recon_output.ply"), recon, (255, 0, 0), 2)

spline_estimator.UpdateCameraPoses(recon, True)
# calculate reprojection error per view
reproj_errors = []
for v_id in sorted(recon.ViewIds):
    view = recon.View(v_id)
    t_ns = view.GetTimestamp()*1e9 
    error = 0

    I = cv2.imread(os.path.join(base_path,run,str(int(t_ns))+".png"))
    for t_id in view.TrackIds():
        feat = view.GetFeature(t_id)

        pt4 = recon.Track(t_id).Point()
        pt3 = np.expand_dims(pt4[0:3]/pt4[3],1)
        pose_vec = spline_estimator.GetCameraPose(int(t_ns + 1e9*spline_estimator.GetRSLineDelay()*feat.point[0]))
        R_w_c = R.from_quat(pose_vec[:4])
        p_w_c = np.expand_dims(pose_vec[4:],1)   

        pt_in_cam = camera.CameraIntrinsics().CameraToImageCoordinates(R_w_c.as_matrix().T @ (pt3 - p_w_c))
        I = cv2.drawMarker(I, (int(feat.point[0]),int(feat.point[1])), (0,255,0),cv2.MARKER_CROSS)
        I = cv2.drawMarker(I, (int(pt_in_cam[0]),int(pt_in_cam[1])), (0,255,255),cv2.MARKER_CROSS)
        repro_err = cv2.norm(feat.point - pt_in_cam)

        #if repro_err > 5.0:
        #    recon.RemoveTrack(t_id)
        #    # print("Removing Track. Reprojection error too high: {:.2f}".format(repro_err))
        error += repro_err
    cv2.imshow("Reprojection from Spline",I)
    cv2.waitKey(telemetry.telemetry["camera_fps"])
    reproj_errors.append(error/len(view.TrackIds()))
print("Mean RS reprojection_error after optim: ",np.sum(reproj_errors)/len(reproj_errors))

optim_gyro_bias = spline_estimator.GetGyroBias(int(5*1e9))
optim_accl_bias = spline_estimator.GetAcclBias(int(5*1e9))
print("Optimized gyro bias: ", optim_gyro_bias) 
print("Optimized acclbias: ", optim_accl_bias)

pvi.WriteSpline(spline_estimator, os.path.join(base_path,"spline_recon_"+run+".spline"))
pt.io.WriteReconstruction(recon, os.path.join(base_path,"spline_recon_"+run+".recon"))
pt.io.WritePlyFile(os.path.join(base_path,"spline_output_"+run+".ply"), recon, (255, 0, 0), 2)


update_dataset_patches(os.path.join(base_path,dpvo_res), recon, view_ids_to_pt_ids)

import matplotlib.pyplot as plt

accl_spline = []
accl_imu = []
accl_bias = []
gyro_spline = []
gyro_imu = []
gyro_bias = []

pos_spline = []
pos_sfm = []
q_spline = []
q_sfm = []
t = []

for v_id in sorted(recon.ViewIds):

    T_w_c = spline_estimator.GetCameraPose(int(recon.View(v_id).GetTimestamp()*1e9))
    pos_spline.append(T_w_c[4:])
    pos_sfm.append(recon.View(v_id).Camera().GetPosition())
    cam_q = T_w_c[0:4]
    q_spline.append(cam_q)
    q_sfm.append(R.from_matrix(recon.View(v_id).Camera().GetOrientationAsRotationMatrix().T).as_quat())

for idx, t_ns in enumerate(sorted(accl_tns)):


    accl_b = spline_estimator.GetAccelerationBody(t_ns)
    vel_b = spline_estimator.GetAngularVelocity(t_ns)

    ba = spline_estimator.GetAcclBias(t_ns)
    gb = spline_estimator.GetGyroBias(t_ns)

    accl_spline.append(accl_b)
    accl_imu.append(accl_meas[t_ns])
    accl_bias.append(ba)

    gyro_spline.append(vel_b)
    gyro_imu.append(gyro_meas[t_ns])
    gyro_bias.append(gb)

accl_spline_np = np.asarray(accl_spline)
accl_imu_np = np.asarray(accl_imu)
accl_bias_np = np.asarray(accl_bias)
gyro_spline_np = np.asarray(gyro_spline)
gyro_imu_np = np.asarray(gyro_imu)
gyro_bias_np = np.asarray(gyro_bias)
t_np = np.asarray(t)
skip = 4

labels = ['spline x', 'imu y', 
        'spline y', 'imu y', 
        'spline z', 'imu z'] 

fig, ax = plt.subplots(2,1)
ax[0].set_title("Accelerometer - Spline value vs Measurements")
ax[0].plot(accl_spline_np[0:-1:skip,0], 'r')
ax[0].plot(accl_imu_np[0:-1:skip,0], 'r--')
ax[0].plot(accl_spline_np[0:-1:skip,1], 'g')
ax[0].plot(accl_imu_np[0:-1:skip,1], 'g--')
ax[0].plot(accl_spline_np[0:-1:skip,2], 'b')
ax[0].plot(accl_imu_np[0:-1:skip,2], 'b--')
ax[0].set_xlabel("measurement")
ax[0].set_ylabel("m/s2")

ax[1].set_title("Gyroscope - Spline value vs Measurements")
ax[1].plot(gyro_spline_np[0:-1:skip,0], 'r', label="spline rot vel x")
ax[1].plot(gyro_imu_np[0:-1:skip,0], 'r--', label="gyro rot vel x")
ax[1].plot(gyro_spline_np[0:-1:skip,1], 'g', label="spline rot vel y")
ax[1].plot(gyro_imu_np[0:-1:skip,1], 'g--', label="gyro rot vel y")
ax[1].plot(gyro_spline_np[0:-1:skip,2], 'b', label="spline rot vel z")
ax[1].plot(gyro_imu_np[0:-1:skip,2], 'b--', label="gyro rot vel z")
ax[1].set_xlabel("measurement")
ax[1].set_ylabel("rad/s")
fig.legend(ax, labels=labels, loc="upper right", borderaxespad=0.1) 
plt.show()

fig, ax = plt.subplots(2,1)
ax[0].set_title("Accelerometer bias")
ax[0].plot(accl_bias_np[0:-1:skip,0], 'r')
ax[0].plot(accl_bias_np[0:-1:skip,1], 'g')
ax[0].plot(accl_bias_np[0:-1:skip,2], 'b')
ax[0].set_xlabel("time")
ax[0].set_ylabel("m/s2")

ax[1].set_title("Gyroscope bias")
ax[1].plot(gyro_bias_np[0:-1:skip,0], 'r', label="bias x")
ax[1].plot(gyro_bias_np[0:-1:skip,1], 'g', label="bias y")
ax[1].plot(gyro_bias_np[0:-1:skip,2], 'b', label="bias z")
ax[1].set_xlabel("time")
ax[1].set_ylabel("rad/s")
plt.show()

pos_sfm_np = np.array(pos_sfm)
pos_spline_np = np.array(pos_spline)

q_sfm_np = np.array(q_sfm)
q_spline_np = np.array(q_spline)

skip = 1
fig, ax = plt.subplots(2,1)
ax[0].set_title("Camera position")
ax[0].plot(pos_spline_np[0:-1:skip,0], 'r')
ax[0].plot(pos_spline_np[0:-1:skip,1], 'g')
ax[0].plot(pos_spline_np[0:-1:skip,2], 'b')
ax[0].plot(pos_sfm_np[0:-1:skip,0], 'r--')
ax[0].plot(pos_sfm_np[0:-1:skip,1], 'g--')
ax[0].plot(pos_sfm_np[0:-1:skip,2], 'b--')
ax[0].set_xlabel("time")
ax[0].set_ylabel("m/s2")

ax[1].set_title("Camera orientation")
ax[1].plot(q_spline_np[0:-1:skip,0], 'r')
ax[1].plot(q_spline_np[0:-1:skip,1], 'g')
ax[1].plot(q_spline_np[0:-1:skip,2], 'b')
ax[1].plot(q_spline_np[0:-1:skip,3], 'c')
ax[1].plot(q_sfm_np[0:-1:skip,0], 'r--')
ax[1].plot(q_sfm_np[0:-1:skip,1], 'g--')
ax[1].plot(q_sfm_np[0:-1:skip,2], 'b--')
ax[1].plot(q_sfm_np[0:-1:skip,3], 'c--')

ax[1].set_xlabel("time")
ax[1].set_ylabel("m/s2")
plt.show()


# do some visualization


