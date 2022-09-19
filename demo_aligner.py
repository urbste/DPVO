import cv2
import numpy as np
import os
import csv
import pickle
from dpvo.lietorch.groups import SE3
import torch
from scipy.spatial.transform import Rotation as R
from dpvo.dpvo_traj_aligner import DPVOAligner
from dpvo.config import cfg

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def get_cam_matrix(calib_file):

    calib = np.loadtxt(calib_file, delimiter=" ")

    fx, fy, cx, cy, Iw, Ih = calib[:6]

    Iw, Ih = int(Iw), int(Ih)

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    distortion = calib[6:]
    return K, distortion, [Iw-Iw%16, Ih-Ih%16]

@torch.no_grad()
def run(cfg, network, match_kf_csv, image1dir, image2dir, calib1, calib2, viz=False, timeit=False):

    slam = None

    K1, dist1, Iwh1 = get_cam_matrix(calib1)
    K2, dist2, Iwh2 = get_cam_matrix(calib2)

    # read matching keyframe timestamps csv file
    timestamp_pairs = []
    with open(match_kf_csv, "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            timestamp_pairs.append(row)
    
    slam = DPVOAligner(cfg, network, ht=Iwh1[1], wd=Iwh1[0], viz=viz)

    # open file to save transformations

    out_dict = {}
    for t_ns_pair in timestamp_pairs:

        image1 = cv2.imread(os.path.join(image1dir, t_ns_pair[0]+".png"))
        image2 = cv2.imread(os.path.join(image2dir, t_ns_pair[1]+".png"))

        #if len(calib) > 4:
        #    image1 = cv2.undistort(image1, K1, dist1[4:])
        #    image2 = cv2.undistort(image2, K2, dist2[4:])

        h, w, _ = image1.shape
        image1 = image1[:h-h%16, :w-w%16]
        image2 = image2[:h-h%16, :w-w%16]

        image1 = torch.from_numpy(image1).permute(2,0,1).cuda()
        image2 = torch.from_numpy(image2).permute(2,0,1).cuda()

        intrinsics1 = torch.tensor([K1[0,0], K1[1,1], K1[0,2], K1[1,2]]).cuda()
        intrinsics2 = torch.tensor([K2[0,0], K2[1,1], K2[0,2], K2[1,2]]).cuda()

        # run VO on both images
        slam(0, image1, intrinsics1, int(t_ns_pair[0]))
        slam(1, image2, intrinsics2, int(t_ns_pair[1]))

        for _ in range(12):
            slam.update()

        T0 = SE3(slam.poses_[0,:])
        T1 = SE3(slam.poses_[1,:])

        T01 = T1 * T0.inv()

        t01 = T01.translation().cpu().numpy()[:3]
        q01 = R.from_matrix(T01.matrix()[:3,:3].cpu().numpy()).as_quat()

        out_dict[int(t_ns_pair[0])] = {
           "t_ns_2" : int(t_ns_pair[1]),
           "t01" : t01.tolist(), 
           "q01" : q01.tolist()
        }
        print("Estimated relative pose between {} and {}".format(t_ns_pair[0], t_ns_pair[1]))
        print("Translation: {:.3f},{:.3f},{:.3f}.".format(t01[0],t01[1],t01[2]))
        print("Quaternion: {:.3f},{:.3f},{:.3f},{:.3f}.".format(q01[0],q01[1],q01[2],q01[3]))

        slam.reset()

    a_file = open(args.savefile, "wb")
    pickle.dump(out_dict, a_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--image1dir', type=str, default="/media/Data/Sparsenet/TestAlignment/bike1_trail1_linear_imgs")
    parser.add_argument('--image2dir', type=str, default="/media/Data/Sparsenet/TestAlignment/bike2_trail1_linear_imgs")
    parser.add_argument('--match_kfs_csv', type=str, default="/media/Data/Sparsenet/TestAlignment/maching_kf_tns.txt")
    parser.add_argument('--calib1', type=str, default="calib/gopro9_linear.txt")
    parser.add_argument('--calib2', type=str, default="calib/gopro9_linear.txt")

    parser.add_argument('--config', default="config/medium.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--savefile', type=str, default='/media/Data/Sparsenet/TestAlignment/relative_trafos.dict')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    run(cfg, args.network, args.match_kfs_csv, args.image1dir, 
        args.image2dir, args.calib1, args.calib2, args.viz, args.timeit)

