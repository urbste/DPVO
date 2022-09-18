import cv2
import numpy as np
import glob
import os.path as osp
import os
from dpvo.lietorch.groups import SE3
import torch
from multiprocessing import Process, Queue

from dpvo.utils import Timer
from dpvo.dpvo_traj_aligner import DPVOAligner
from dpvo.config import cfg
from dpvo.stream import image_align_stream

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, match_kf_csv, image1dir, image2dir, calib, viz=False, timeit=False):

    slam = None
    queue = Queue(maxsize=2)

    reader = Process(target=image_align_stream, args=(queue, match_kf_csv, image1dir, image2dir, calib))

    reader.start()

    while 1:
        (idx, image, intrinsics, t_ns) = queue.get()
        if idx < 0: break
        print("idx",idx)
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVOAligner(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=timeit):
            slam(idx, image, intrinsics, t_ns)

    for _ in range(12):
        slam.update()

    reader.join()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--image1dir', type=str, default="/media/Data/Sparsenet/TestAlignment/bike1_trail1_linear_imgs")
    parser.add_argument('--image2dir', type=str, default="/media/Data/Sparsenet/TestAlignment/bike2_trail1_linear_imgs")
    parser.add_argument('--match_kfs_csv', type=str, default="/media/Data/Sparsenet/TestAlignment/maching_kf_tns.txt")
    parser.add_argument('--calib', type=str, default="calib/gopro9_linear.txt")
    parser.add_argument('--config', default="config/medium.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--savefile', type=str, default='/media/Data/Sparsenet/TestAlignment/relative_trafos.txt')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)


    result = run(cfg, args.network, args.match_kfs_csv, args.image1dir, args.image2dir, args.calib, 
        args.viz, args.timeit)

    all_poses, kf_poses, tstamps, image_stamps_ns, patches, indices, ii, jj, kk, intrinsics = result
    rel_pose = SE3(torch.tensor(all_poses[1,:])) * SE3(torch.tensor(all_poses[0,:])).inv()
    np.save(args.savefile, rel_pose.matrix().numpy())
    print("rel pose", rel_pose.matrix())
    # np.savez(args.savefile, 
    #     name1=all_poses, 
    #     name2=kf_poses,
    #     name3=tstamps, 
    #     name4=image_stamps_ns, 
    #     name5=patches,
    #     name6=indices, 
    #     name7=ii, 
    #     name8=jj, 
    #     name9=kk,
    #     name10=intrinsics)
        

