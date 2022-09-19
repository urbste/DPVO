import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
import natsort
from csv import reader


def image_stream(queue, imagedir, calib, stride, skip=0, start_end_t_ns=[0,0]):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy, Iw, Ih = calib[:6]

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
        image = cv2.resize(image, (int(Iw), int(Ih)))

        if len(calib) > 6:
            image = cv2.undistort(image, K, calib[6:])

        intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics, t_ns))

    queue.put((-1, image, intrinsics, t_ns))

def video_stream(queue, imagedir, calib, stride, skip=0, start_end_t_ns=[0,0]):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy, Iw, Ih = calib[:6]

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

        if len(calib) > 6:
            image = cv2.undistort(image, K, calib[6:])

        # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (int(Iw), int(Ih)))
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((idx, image, intrinsics, t_ns))

        idx += 1

    queue.put((-1, image, intrinsics, -1))
    cap.release()

