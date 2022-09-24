import pyhbst
import natsort
import cv2
import os
import numpy as np

class PlaceRecognition:
    def __init__(self, img_folder, kf_timestamps, img_size=(640,480)):
        # construct a tree supporting some descriptor size (64,128,256,512)
        self.tree = pyhbst.BinarySearchTree256() # descriptor size 256-bit

        self.orb = cv2.ORB_create(1000, 1.2, 2, 32, 0, 2, 0, 25, 20)
        # Initiate FAST detector
        # Initiate FAST object with default values
        #self.fast = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True)
        # Initiate BRIEF extractor
        #self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(64, False)

        self.img_size = img_size
        self.idx_to_t_ns1 = {}
        self.last_idx_database = 0

        self.kf_t_ns_1 = kf_timestamps

        print("Adding images to tree. Might take some time.")
        imglist = natsort.natsorted(os.listdir(img_folder))
        idx = 0
        for i_name in imglist:
            t_ns = int(os.path.splitext(i_name)[0])
            if t_ns not in self.kf_t_ns_1:
                continue

            I = self.load_image(os.path.join(img_folder, i_name))
            kpts, desc = self.get_features(I)

            self.tree.add(kpts, desc, idx, pyhbst.SplitEven)
            self.idx_to_t_ns1[idx] = t_ns
            if idx % 100 == 0:
                print("Added {}/{} images to tree.".format(idx, len(kf_timestamps)))
                print("num kpts", len(kpts))

            idx += 1

        print("Place recognizer initialized.")
        self.last_idx_database = idx

    def load_image(self, path):
        return cv2.resize(cv2.imread(path,0), self.img_size)

    def get_features(self, img):
        kpts, desc = self.orb.detectAndCompute(img, None)
        #kpts = self.fast.detect(img, None)
        #kpts, desc = self.brief.compute(img, kpts, None)
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
        t_ns1_to_t_ns2 = []
        imglist = natsort.natsorted(os.listdir(img_folder))
        idx = 0
        for i_name in imglist:
            t_ns = int(os.path.splitext(i_name)[0])

            if t_ns not in kf_timestamps:
                continue

            I = self.load_image(os.path.join(img_folder, i_name))
            kpts, desc = self.get_features(I)
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
                    t_ns1_to_t_ns2.append([ref_t_ns, t_ns])

                    if debug:
                        Iref = self.load_image(os.path.join(img_folder, str(ref_t_ns)+".png"))
                        cv2.imshow("matching kfs", np.concatenate([Iref, I],1))
                        cv2.waitKey(0)
            idx += 1
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

kfs1_tns = dataset1["frametimes_ns"]
kfs2_tns = dataset2["frametimes_ns"]

# create database with first dataset
matcher = PlaceRecognition(os.path.join(base_path,"run1"), kfs1_tns)

matcher.localize_img_folder(
    os.path.join(base_path,"run2"), kfs2_tns, min_matches=30, hamming_dist=45, debug=True)

