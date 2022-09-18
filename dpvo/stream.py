import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
import natsort
from csv import reader

def image_align_stream(queue, match_idxs_csv, imagedir1, imagedir2, calib):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    # read matching keyframe timestamps csv file
    timestamp_pairs = []
    with open(match_idxs_csv, "r") as f:
        csv_reader = reader(f)
        for row in csv_reader:
            timestamp_pairs.append(row)
    
    for t_ns_pair in timestamp_pairs:
        image1 = cv2.imread(os.path.join(imagedir1, t_ns_pair[0]+".png"))
        image2 = cv2.imread(os.path.join(imagedir2, t_ns_pair[1]+".png"))
        if len(calib) > 4:
            image1 = cv2.undistort(image1, K, calib[4:])
            image2 = cv2.undistort(image2, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image1.shape
        image1 = image1[:h-h%16, :w-w%16]
        image2 = image2[:h-h%16, :w-w%16]

        queue.put((0, image1, intrinsics, int(t_ns_pair[0])))
        queue.put((1, image2, intrinsics, int(t_ns_pair[1])))

    queue.put((-1, image1, intrinsics, int(t_ns_pair[0])))


def image_stream(queue, imagedir, calib, stride, skip=0, start_end_t_ns=[0,0]):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = natsort.natsorted(os.listdir(imagedir))[skip::stride]

    for t, imfile in enumerate(image_list):
        t_ns = int(os.path.splitext(os.path.basename(imfile))[0])

        if t_ns < start_end_t_ns[0]:
            continue
        if t_ns > start_end_t_ns[1] and start_end_t_ns[1] > start_end_t_ns[0]:
            break

        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics, t_ns))

    queue.put((-1, image, intrinsics, t_ns))

def video_stream(queue, imagedir, calib, stride, skip=0, start_end_t_ns=[0,0]):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy
    intrinsics = np.array([fx, fy, cx, cy])

    cap = cv2.VideoCapture(imagedir)

    idx = 0

    #for _ in range(skip):
    #    ret, image = cap.read()
    invalid_images = 0
    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                invalid_images += 1
                if invalid_images > 400:
                    break
                continue
        if not ret:
            invalid_images += 1
            if invalid_images > 400:
                break
            continue
        t_ns = int(1e6*cap.get(cv2.CAP_PROP_POS_MSEC))

        if t_ns < start_end_t_ns[0]:
            continue
        if t_ns > start_end_t_ns[1] and start_end_t_ns[1] > start_end_t_ns[0]:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (480, 270))
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((idx, image, intrinsics, t_ns))

        idx += 1

    queue.put((-1, image, intrinsics, -1))
    cap.release()

