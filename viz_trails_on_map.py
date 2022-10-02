import os
from tracemalloc import start
import pymap3d
import numpy as np
import pyvisfm as pvi
import natsort
from utils import load_dataset
from telemetry_converter import TelemetryImporter

def get_lon_lat_dataset(base_path, run, ll0):

    tel_json = run+".json"
    dpvo_res = "dpvo_result_"+run+".npz"
    dataset = load_dataset(
        os.path.join(base_path,dpvo_res),
        os.path.join(base_path,tel_json),
        None, inv_depth_thresh=0.5, 
        scale_with_gps=True, align_with_grav=True, correct_heading=True)
    gps_ned_t = np.array(dataset["gps_local_ned"])

    spline = pvi.SplineTrajectoryEstimator()
    #end_t_ns = spline.GetMaxTimeNs()
    #start_t_ns = spline.GetMinTimeNs()
    
    pvi.ReadSpline(spline, 
        os.path.join(base_path,"spline_recon_"+run+".spline"))
    image_folder = os.listdir(os.path.join(base_path,"run1"))
    timestamps1 = natsort.natsorted(
        [os.path.splitext(os.path.basename(p))[0] for p in image_folder])

    latlonh = []
    for t_ns in timestamps1:
        t_ns_ = int(t_ns)
        #if t_ns_ < start_t_ns or t_ns_ > end_t_ns:
        #    continue
        pose_vec= spline.GetPose(t_ns_)
        ned = pose_vec[4:] + gps_ned_t[0,:]
        lat, lon, h = pymap3d.ned2geodetic(ned[0],ned[1],ned[2],
            ll0[0], ll0[1], ll0[2])
        latlonh.append([lat, lon, h])
    return latlonh

base_path = "/media/Data/Sparsenet/Ammerbach/Links"

tel_importer = TelemetryImporter()
tel_importer.read_gopro_telemetry(os.path.join(base_path,"run1.json"))
ll0 = llh0 = tel_importer.telemetry["gps_llh"][0]

llh1 = np.array(get_lon_lat_dataset(base_path, "run1", ll0))
llh2 = np.array(get_lon_lat_dataset(base_path, "run2", ll0))



import mplleaflet  
import matplotlib.pyplot as plt
plt.plot(llh1[:,0], llh1[:,1], color='red', marker='o', markersize=3, linewidth=2, alpha=0.4)
plt.plot(llh2[:,0], llh2[:,1], color='green', marker='o', markersize=3, linewidth=2, alpha=0.4)
#mplleaflet.display(fig=ax.figure)  # shows map inline in Jupyter but takes up full width
mplleaflet.show(path='trail_on_map.html')  # saves to html file for display below