import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from multiprocessing import Process, Queue

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False, start_end_t=[0,0]):

    slam = None
    queue = Queue(maxsize=8)

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip, start_end_t))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip, start_end_t))

    reader.start()

    while 1:
        (idx, image, intrinsics, t_ns) = queue.get()
        if idx < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

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
    parser.add_argument('--imagedir', type=str, default="/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/JenzigTrailsJune/bike1_trail1_linear.MP4")
    parser.add_argument('--calib', type=str, default="calib/gopro9_linear.txt")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/medium.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--start_t_ns', type=int, default=0)
    parser.add_argument('--end_t_ns', type=int, default=0)
    parser.add_argument('--savefile', type=str, default='')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)


    result = run(cfg, args.network, args.imagedir, args.calib, 
        args.stride, args.skip, args.viz, args.timeit, [args.start_t_ns, args.end_t_ns])

    all_poses, kf_poses, tstamps, image_stamps_ns, patches, indices, ii, jj, kk, intrinsics = result

    np.savez(args.savefile, 
        name1=all_poses, 
        name2=kf_poses,
        name3=tstamps, 
        name4=image_stamps_ns, 
        name5=patches,
        name6=indices, 
        name7=ii, 
        name8=jj, 
        name9=kk,
        name10=intrinsics)
        

