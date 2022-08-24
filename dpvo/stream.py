import os
import cv2
import numpy as np
from multiprocessing import Process, Queue

def image_stream(queue, imagedir, calib, stride, skip=0):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[skip::stride]

    for t, imfile in enumerate(image_list):
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

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(queue, imagedir, calib, stride, skip=0, start_end_ts=[0.,0.]):
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

    t = 0

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
        ts_ns = int(1e6*cap.get(cv2.CAP_PROP_POS_MSEC))
        ts_s = ts_ns*1e-9
        if ts_s < start_end_ts[0]:
            continue
        if ts_s > start_end_ts[1] and start_end_ts[1] > start_end_ts[0]:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (480, 270))
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((ts_ns, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

