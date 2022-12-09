import pyhbst
import natsort
import cv2
import os
import numpy as np

class PlaceRecognition:
    def __init__(self, img_folder, kf_timestamps, img_size=(640,480)):
        # construct a tree supporting some descriptor size (64,128,256,512)
        self.tree = pyhbst.BinarySearchTree256() # descriptor size 256-bit

        #self.orb = cv2.ORB_create(5000, 1.2, 4, 32, 0, 2, 0, 25, 10)
        # Initiate FAST detector
        # Initiate FAST object with default values
        self.fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
        # Initiate BRIEF extractor
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(32, False)

        self.img_size = img_size
        self.idx_to_t_ns1 = {}
        self.last_idx_database = 0

        self.kf_t_ns_1 = kf_timestamps
        self.img_folder_ref = img_folder
        self.min_nr_keypts = 600

        print("Adding images to tree. Might take some time.")
        imglist = natsort.natsorted(os.listdir(img_folder))
        idx = 0
        for i_name in imglist:
            t_ns = int(os.path.splitext(i_name)[0])
            if t_ns not in self.kf_t_ns_1:
                continue

            I = self.load_image(os.path.join(img_folder, i_name))
            kpts, desc = self.get_features(I)

            if len(kpts) < self.min_nr_keypts:
                continue
            self.tree.add(kpts, desc, idx, pyhbst.SplitEven)
            self.idx_to_t_ns1[idx] = t_ns
            if idx % 1 == 0:
                print("Added {}/{} images to tree.".format(idx, len(kf_timestamps)))
                print("num kpts", len(kpts))

            idx += 1

        print("Place recognizer initialized.")
        self.last_idx_database = idx

    def load_image(self, path):
        return cv2.resize(cv2.imread(path,0), self.img_size)

    def get_features(self, img):
        #kpts, desc = self.orb.detectAndCompute(img, None)
        kpts = self.fast.detect(img, None)
        kpts, desc = self.brief.compute(img, kpts, None)
        kpts_list = [list(kpt.pt) for kpt in kpts]

        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #img = cv2.drawKeypoints(img, kpts, img, color=(255,0,0))
        #cv2.imshow("kpts",img)
        #cv2.waitKey(0)
        return kpts_list, desc.tolist()

    def localize_img_folder(self, img_folder, kf_timestamps, min_matches=60, hamming_dist=25, debug=False):

        print("Localizing images to tree. Might take some time.")
        print("Min number of matches for successfull recognition: {} with hamming distance thresh: {}".format(
            min_matches, hamming_dist))

        start_idx = self.last_idx_database + 1
        self.idx_to_t_ns2 = {}
        t_tef_to_query = {}
        imglist = natsort.natsorted(os.listdir(img_folder))
        idx = 0
        for i_name in imglist:
            t_ns = int(os.path.splitext(i_name)[0])

            if t_ns not in kf_timestamps:
                continue

            I = self.load_image(os.path.join(img_folder, i_name))
            kpts, desc = self.get_features(I)
            if len(kpts) < self.min_nr_keypts:
                print("Skip not enough keypoints")
                continue
            #cv2.imshow("current image", I)
            #cv2.waitKey(10)

            scores = self.tree.getScorePerImage(kpts, desc, start_idx + idx, True, hamming_dist)
            for s in scores:
                print("Number of matches: ", s.number_of_matches)
                if s.number_of_matches < min_matches:
                    break
                else:
                    ref_id = s.identifier_reference
                    ref_t_ns = self.idx_to_t_ns1[ref_id]
                    if ref_t_ns not in t_tef_to_query:
                        t_tef_to_query[ref_t_ns] = {"query_ts": [t_ns], "nr_matches": [s.number_of_matches]}
                    else:
                        t_tef_to_query[ref_t_ns]["query_ts"].append(t_ns)
                        t_tef_to_query[ref_t_ns]["nr_matches"].append(s.number_of_matches)
                    if debug:
                        Iref = self.load_image(os.path.join(self.img_folder_ref, str(ref_t_ns)+".png"))
                        cv2.imshow("matching kfs", np.concatenate([I, Iref],1))
                        cv2.waitKey(1)
                    break
            idx += 1
        return t_tef_to_query

    def __del__(self):
        self.tree.clear(True)


from utils import load_dataset
from telemetry_converter import TelemetryImporter
base_path = "/media/Data/Sparsenet/Ammerbach/Links"

# load telemetry
telemetry = TelemetryImporter()
telemetry.read_gopro_telemetry(os.path.join(base_path,"run1.json"))
llh0 = telemetry.telemetry["gps_llh"][0]
dataset1 = load_dataset(
    os.path.join(base_path,"dpvo_result_run1.npz"),
    os.path.join(base_path,"run1.json"),
    llh0, inv_depth_thresh=0.5, 
    scale_with_gps=False, align_with_grav=False, correct_heading=False)

dataset2 = load_dataset(
    os.path.join(base_path,"dpvo_result_run2.npz"),
    os.path.join(base_path,"run2.json"),
    llh0, inv_depth_thresh=0.5, 
    scale_with_gps=False, align_with_grav=False, correct_heading=False)

kfs1_tns = dataset1["frametimes_slam_ns"]
kfs1_tns = kfs1_tns[::2]
print("Adding {} kfs".format(len(kfs1_tns)))
kfs2_tns = dataset2["frametimes_slam_ns"]

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
t_tef_to_query_max = {}
# add first and last frame
t_tef_to_query_max[kfs1_tns[0]] = kfs2_tns[0]
t_tef_to_query_max[kfs1_tns[-1]] = kfs2_tns[-1]

# create database with first dataset
#kfs2_tns = list(const_to_base_trail.keys())
#kfs1_tns = [const_to_base_trail[o]["time_base"] for o in const_to_base_trail]
matcher = PlaceRecognition(os.path.join(base_path,"run1"), kfs1_tns)


t_tef_to_query = matcher.localize_img_folder(
    os.path.join(base_path,"run2"), kfs2_tns, min_matches=30, hamming_dist=30, debug=True)

# only get the match with the largest number of features
for t_ref in t_tef_to_query:
    nr_matches = sorted(t_tef_to_query[t_ref]["nr_matches"], reverse=True)[0]
    idx = t_tef_to_query[t_ref]["nr_matches"].index(nr_matches)
    query_t_ns = t_tef_to_query[t_ref]["query_ts"][idx]

    t_tef_to_query_max[t_ref] = query_t_ns

    Iref = matcher.load_image(os.path.join(base_path,"run1", str(t_ref)+".png"))
    Iquery = matcher.load_image(os.path.join(base_path,"run2", str(query_t_ns)+".png"))
    cv2.imshow("matching kfs", np.concatenate([Iref, Iquery],1))
    cv2.waitKey(100)


import json
with open(os.path.join(base_path, "matching_kfs_run1_run2.json"), "w") as f:
    json.dump(t_tef_to_query_max, f)