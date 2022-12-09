import cv2
import numpy as np
import os, json
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
    Iw_new = Iw-Iw%16
    Ih_new = Ih-Ih%16
    K = np.eye(3)
    K[0,0] = fx * Iw_new/Iw 
    K[0,2] = cx * Iw_new/Iw 
    K[1,1] = fy * Ih_new/Ih 
    K[1,2] = cy * Ih_new/Ih 

    distortion = calib[6:]
    return K, distortion, [Iw-Iw%16, Ih-Ih%16]

@torch.no_grad()
def run(cfg, network, match_kf_json, patches_spline, image1dir, image2dir, calib1, calib2, viz=False, timeit=False):

    slam = None

    K1, dist1, Iwh1 = get_cam_matrix(calib1)
    K2, dist2, Iwh2 = get_cam_matrix(calib2)

    # read matching keyframe timestamps csv file
    timestamp_pairs = []
    with open(match_kf_json, "r") as f:
        ref_to_query_constrain = json.load(f)

    data = np.load(patches_spline)
    patches = data["patches"]
    frametimes_ns = data["image_tstamps"]



    slam = DPVOAligner(cfg, network, ht=Iwh1[1], wd=Iwh1[0], viz=viz)
    # open file to save transformations

    out_dict = {}
    for t_ns_ref in ref_to_query_constrain:
        t_ns_ref_ = int(t_ns_ref)
        t_ns_query = ref_to_query_constrain[t_ns_ref]
        image1 = cv2.resize(cv2.imread(os.path.join(image1dir, str(t_ns_ref_)+".png")), Iwh1)
        image2 = cv2.resize(cv2.imread(os.path.join(image2dir, str(t_ns_query)+".png")), Iwh2)

        patch_ref = patches[[frametimes_ns.tolist().index(t_ns_ref_)]]
        slam.set_patches(patch_ref)

        #if len(calib) > 4:
        #    image1 = cv2.undistort(image1, K1, dist1[4:])
        #    image2 = cv2.undistort(image2, K2, dist2[4:])

        h, w, _ = image1.shape

        image1 = torch.from_numpy(image1).permute(2,0,1).cuda()
        image2 = torch.from_numpy(image2).permute(2,0,1).cuda()

        intrinsics1 = torch.tensor([K1[0,0], K1[1,1], K1[0,2], K1[1,2]]).cuda()
        intrinsics2 = torch.tensor([K2[0,0], K2[1,1], K2[0,2], K2[1,2]]).cuda()

        # run VO on both images
        slam(0, image1, intrinsics1, int(t_ns_ref))
        slam(1, image2, intrinsics2, int(t_ns_query))

        for _ in range(12):
            slam.update()

        T0 = SE3(slam.poses_[0,:])
        T1 = SE3(slam.poses_[1,:])

        T01 = T1 * T0.inv()

        t01 = T01.translation().cpu().numpy()[:3]
        q01 = R.from_matrix(T01.matrix()[:3,:3].cpu().numpy()).as_quat()

        out_dict[t_ns_ref] = {
           "t_ns_query" : t_ns_query,
           "t01" : t01.tolist(), 
           "q01" : q01.tolist()
        }
        print("Estimated relative pose between {} and {}".format(t_ns_ref, t_ns_query))
        print("Translation: {:.3f},{:.3f},{:.3f}.".format(t01[0],t01[1],t01[2]))
        print("Quaternion: {:.3f},{:.3f},{:.3f},{:.3f}.".format(q01[0],q01[1],q01[2],q01[3]))

        slam.reset()

    a_file = open(args.savefile, "wb")
    pickle.dump(out_dict, a_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--patches_run1_spline', type=str, 
        default="/media/Data/Sparsenet/Ammerbach/Links/dpvo_result_run1_spline.npz")

    parser.add_argument('--image1dir', type=str, default="/media/Data/Sparsenet/Ammerbach/Links/run1")
    parser.add_argument('--image2dir', type=str, default="/media/Data/Sparsenet/Ammerbach/Links/run2")
    parser.add_argument('--match_kfs', type=str, default="/media/Data/Sparsenet/Ammerbach/Links/matching_kfs_run1_run2.json")
    parser.add_argument('--calib1', type=str, default="calib/gopro9_linear.txt")
    parser.add_argument('--calib2', type=str, default="calib/gopro9_linear.txt")

    parser.add_argument('--config', default="config/medium.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--savefile', type=str, default='/media/Data/Sparsenet/Ammerbach/Links/relative_trafos_run1_run2.dict')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    run(cfg, args.network, args.match_kfs, args.patches_run1_spline, args.image1dir, 
        args.image2dir, args.calib1, args.calib2, args.viz, args.timeit)

