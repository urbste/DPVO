import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from gps_converter import ECEFToLLA, ECEFtoENU, LLAtoECEF

class TelemetryImporter:
    ''' TelemetryImporter

    '''
    def __init__(self):    
        self.ms_to_sec = 1e-3
        self.us_to_sec = 1e-6
        self.ns_to_sec = 1e-9

        self.telemetry = {}

    def _remove_seconds(self, accl, gyro, timestamps_ns, skip_seconds):
        skip_ns = skip_seconds / self.ns_to_sec

        ds = timestamps_ns[1] - timestamps_ns[0]
        nr_remove = round(skip_ns / ds)

        accl = accl[nr_remove:len(timestamps_ns) - nr_remove]
        gyro = gyro[nr_remove:len(timestamps_ns) - nr_remove]
        timestamps_ns = timestamps_ns[nr_remove:len(timestamps_ns) - nr_remove]

        return accl, gyro, timestamps_ns
    '''
    json_data : dict
        content of json file
    skip_seconds : float
        How many seconds to cut from beginning and end of stream
    '''
    def read_gopro_telemetry(self, path_to_json, skip_seconds=0.0):

        json_file = open(path_to_json, 'r')
        json_data = json.load(json_file)

        accl = []
        gyro  = []
        cori = []
        gravity = []
        gps_ecef = []
        gps_llh = []
        gps_prec = []
        img_timestamps_ns = []
        gps_timestamps_ns = []
        timestamps_ns = []
        # acceleration at imu rate
        for a in json_data['1']['streams']['ACCL']['samples']:
            timestamps_ns.append(a['cts'] * self.ms_to_sec / self.ns_to_sec)
            accl.append([a['value'][1], a['value'][2], a['value'][0]])
        # gyroscope at imu rate
        for g in json_data['1']['streams']['GYRO']['samples']:
            gyro.append([g['value'][1], g['value'][2], g['value'][0]])
        # image orientation at framerate
        for c in json_data['1']['streams']['CORI']['samples']:
            # order w,x,z,y https://github.com/gopro/gpmf-parser/issues/100#issuecomment-656154136
            w, x, z, y = c['value'][0], c['value'][1], c['value'][2], c['value'][3]
            cori.append([x, y, z, w])
            img_timestamps_ns.append(c['cts'] * self.ms_to_sec / self.ns_to_sec)
        
        # gravity vector in camera coordinates at framerate
        for g in json_data['1']['streams']['GRAV']['samples']:
            gravity.append([g['value'][0], g['value'][1], g['value'][2]])
        
        # GPS
        for g in json_data["1"]["streams"]["GPS5"]["samples"]:
            gps_timestamps_ns.append(g['cts'] * self.ms_to_sec / self.ns_to_sec)
            lat, long, alt = g["value"][0], g["value"][1], g["value"][2]
            x,y,z = LLAtoECEF(lat, long, alt)
            gps_llh.append([lat,long,alt])
            gps_ecef.append([x,y,z])
            gps_prec.append(g["precision"])

        camera_fps = json_data['frames/second']
        if skip_seconds != 0.0:
            accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)

        accl = accl[0:len(timestamps_ns)]
        gyro = gyro[0:len(timestamps_ns)]
        cori = cori[0:len(img_timestamps_ns)]
        gravity = gravity[0:len(img_timestamps_ns)]

        self.telemetry["accelerometer"] = accl
        self.telemetry["gyroscope"] = gyro
        self.telemetry["timestamps_ns"] = timestamps_ns
        self.telemetry["camera_fps"] = camera_fps
        self.telemetry["camera_orientation"] = cori
        self.telemetry["gravity_vector"] = gravity # same timestamps as cori

        self.telemetry["image_timestamps_ns"] = img_timestamps_ns
        self.telemetry["gps_ecef_coords"] = gps_ecef
        self.telemetry["gps_llh"] = gps_llh
        self.telemetry["gps_precision"] = gps_prec
        self.telemetry["gps_timestamps"] = gps_timestamps_ns

    def read_pilotguru_telemetry(self, path_to_accl_json, path_to_gyro_json, path_to_cam_json, skip_seconds=0.0):
        accl_json_file = open(path_to_accl_json, 'r')
        accl_data = json.load(accl_json_file)
        gyro_json_file = open(path_to_gyro_json, 'r')
        gyro_data = json.load(gyro_json_file)       
        cam_json_file = open(path_to_cam_json, 'r')
        cam_data = json.load(cam_json_file)

        accl = []
        gyro  = []
        timestamps_ns = []
        # our timestamps should always start at zero for the camera, so we normalize here
        cam_t0 = cam_data['frames'][0]['time_usec']

        # in addition find out which of the two sensors runs faster (if)
        accl_ps = 1./ ((accl_data['accelerations'][1]['time_usec'] - accl_data['accelerations'][0]['time_usec'])*self.us_to_sec)
        gyro_ps = 1./ ((gyro_data['rotations'][1]['time_usec'] - gyro_data['rotations'][0]['time_usec'])*self.us_to_sec)

        if accl_ps > gyro_ps:
            subsample = int(round(accl_ps / gyro_ps))
            for i in range(0,len(accl_data['accelerations']),subsample):
                timestamps_ns.append(
                    (accl_data['accelerations'][i]['time_usec'] - cam_t0)  * self.us_to_sec / self.ns_to_sec)
                accl.append(
                    [accl_data['accelerations'][i]['x'], 
                     accl_data['accelerations'][i]['y'], 
                     accl_data['accelerations'][i]['z']])
            for g in gyro_data['rotations']:
                gyro.append([g['x'], g['y'], g['z']])
        else:
            subsample = int(round(gyro_ps / accl_ps))
            for a in accl_data['accelerations']:
                accl.append([a['x'], a['y'], a['z']])
            for i in range(0,len(gyro_data['rotations']),subsample):
                timestamps_ns.append(
                    (gyro_data['rotations'][i]['time_usec'] - cam_t0)  * self.us_to_sec / self.ns_to_sec)
                gyro.append([gyro_data['rotations'][i]['x'], gyro_data['rotations'][i]['y'], gyro_data['rotations'][i]['z']])
        camera_fps = 1. / ((cam_data['frames'][1]['time_usec'] - cam_data['frames'][0]['time_usec']) *self.us_to_sec)

        if skip_seconds != 0.0:
            accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)

        accl = accl[0:len(timestamps_ns)]
        gyro = gyro[0:len(timestamps_ns)]

        self.telemetry["accelerometer"] = accl
        self.telemetry["gyroscope"] = gyro
        self.telemetry["timestamps_ns"] = timestamps_ns
        self.telemetry["camera_fps"] = camera_fps
        self.telemetry["camera_orientation"] = []
        self.telemetry["gravity_in_camera"] = []
        self.telemetry["image_timestamps_ns"] = []

    def read_generic_json(self, path_to_json, skip_seconds=0.0):
        json_file = open(path_to_json, 'r')
        json_data = json.load(json_file)

        accl = []
        gyro  = []
        timestamps_ns = []
        for a in json_data['accelerometer']:
            accl.append([a[0], a[1], a[2]])
        for g in json_data['gyroscope']:
            gyro.append([g[0], g[1], g[2]])
        for t in json_data['timestamps_ns']:
            timestamps_ns.append(t)
        
        if skip_seconds != 0.0:
            accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)

        accl = accl[0:len(timestamps_ns)]
        gyro = gyro[0:len(timestamps_ns)]

        self.telemetry["accelerometer"] = accl
        self.telemetry["gyroscope"] = gyro
        self.telemetry["timestamps_ns"] = timestamps_ns
        self.telemetry["camera_fps"] = json_data["camera_fps"] 
        self.telemetry["camera_orientation"] = []
        self.telemetry["gravity_in_camera"] = []
        self.telemetry["image_timestamps_ns"] = []

    def get_camera_quaternions_at_frametimes(self):
        # interpolate camera quaternions to frametimes
        frame_rots = R.from_quat(self.telemetry["camera_orientation"]) 
        frame_times = np.array(self.telemetry["image_timestamps_ns"]) * self.ns_to_sec
        slerp = Slerp(frame_times.tolist(), frame_rots)
        cam_hz = 1 / self.telemetry["camera_fps"]
        start_time = 0.0
        while start_time < frame_times[0]:
            start_time += cam_hz
        interp_frame_times =  np.arange(start_time, frame_times[-1] - cam_hz, cam_hz) 

        interp_rots = slerp(interp_frame_times)
        camera_quaternions = dict(zip( np.round(np.array(interp_frame_times)*1e6,1), interp_rots.as_quat()))

        return camera_quaternions

    def get_gravity_vector_at_times(self, times_ns=None):
        # interpolate camera quaternions to frametimes
        frame_grav = np.array(self.telemetry["gravity_vector"])
        grav_times = np.array(self.telemetry["image_timestamps_ns"]) * self.ns_to_sec

        if times_ns is not None:
            frame_times = np.array(times_ns) * self.ns_to_sec
        else:
            frame_times = np.array(self.telemetry["image_timestamps_ns"]) * self.ns_to_sec

        # find valid interval (interpolate only where we actually have gps measurements)
        start_frame_time_idx = np.where(grav_times[0] < frame_times)[0][0]
        end_frame_time_idx = np.where(grav_times[-1] <= frame_times)[0]
        if not end_frame_time_idx:
            end_frame_time_idx = len(frame_times)-1

        cam_hz = 1 / self.telemetry["camera_fps"]
        if times_ns is not None:
            interp_frame_times = frame_times[start_frame_time_idx:end_frame_time_idx]
        else:
            interp_frame_times = np.round(
            np.arange(
                np.round(frame_times[start_frame_time_idx],2), 
                np.round(frame_times[end_frame_time_idx],2) - cam_hz, cam_hz) ,3).tolist()


        x_interp = np.interp(interp_frame_times, grav_times, frame_grav[:,0])
        y_interp = np.interp(interp_frame_times, grav_times, frame_grav[:,1])
        z_interp = np.interp(interp_frame_times, grav_times, frame_grav[:,2])
        frame_grav_interp = np.stack([x_interp,y_interp,z_interp],1)

        return frame_grav_interp

    def get_gps_pos_at_frametimes(self, img_times_ns=None):

        # interpolate camera gps info at frametimes
        frame_gps_ecef = np.array(self.telemetry["gps_ecef_coords"])
        gps_times = np.array(self.telemetry["gps_timestamps"]) * self.ns_to_sec
        if img_times_ns is not None:
            frame_times = np.array(img_times_ns) * self.ns_to_sec
        else:
            frame_times = np.array(self.telemetry["image_timestamps_ns"]) * self.ns_to_sec
        
        # find valid interval (interpolate only where we actually have gps measurements)
        start_frame_time_idx = np.where(gps_times[0] < frame_times)[0][0]
        end_frame_time_idx = np.where(gps_times[-1] <= frame_times)[0]
        if not end_frame_time_idx:
            end_frame_time_idx = len(frame_times)

        cam_hz = 1 / self.telemetry["camera_fps"]
        if img_times_ns is not None:
            interp_frame_times = frame_times[start_frame_time_idx:end_frame_time_idx]
        else:
            interp_frame_times = np.round(
            np.arange(
                np.round(frame_times[start_frame_time_idx],2), 
                np.round(frame_times[end_frame_time_idx],2) - cam_hz, cam_hz) ,3).tolist()

        x_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,0])
        y_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,1])
        z_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,2])
        prec_interp = np.interp(interp_frame_times, gps_times, self.telemetry["gps_precision"])
        xyz_interp = np.stack([x_interp,y_interp,z_interp],1)

        camera_gps = dict(zip((np.array(interp_frame_times)*1e9).astype(np.int), xyz_interp.tolist()))

        return camera_gps, prec_interp

class TelemetryConverter:
    ''' TelemetryConverter

    '''
    def __init__(self):
        self.output_dict = {}
        self.telemetry_importer = TelemetryImporter()

    def _dump_final_json(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.telemetry_importer.telemetry, f)
    
    def convert_gopro_telemetry_file(self, input_telemetry_json, output_path, skip_seconds=0.0):
        self.telemetry_importer.read_gopro_telemetry(
            input_telemetry_json, skip_seconds=skip_seconds)
        self._dump_final_json(output_path)

    def convert_pilotguru_telemetry_file(self, input_accl_json, input_gyro_json, input_cam_json, output_path, skip_seconds=0.0):
        self.telemetry_importer.read_pilotguru_telemetry(
            input_accl_json, input_gyro_json, input_cam_json, skip_seconds=skip_seconds)
        self._dump_final_json(output_path)