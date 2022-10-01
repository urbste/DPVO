import os, cv2
from natsort import natsorted
from telemetry_converter import TelemetryImporter

def extract_images_from_video_to_folder(video_file, 
    img_savepath, resize_factor, telemetry):

    exact_img_t_ns = []
    if telemetry != '':
        tel = TelemetryImporter()
        tel.read_gopro_telemetry(telemetry)
        exact_img_t_ns = natsorted(tel.telemetry["img_timestamps_ns"])

    cap = cv2.VideoCapture(video_file)

    if not os.path.exists(img_savepath):
        os.makedirs(img_savepath)

    invalid_images = 0
    had_zero = False
    print("Extracting frames")
    img_idx = 0
    while True:
        ret, I = cap.read()
        ts_ns = int(1e6*cap.get(cv2.CAP_PROP_POS_MSEC))
        if not ret:
            invalid_images += 1
            if invalid_images > 100:
                break
            continue
        if had_zero and ts_ns == 0:
            continue

        if not had_zero and ts_ns == 0:
            had_zero = True

        if len(exact_img_t_ns) > 0:
            ts_ns =int(round(exact_img_t_ns[img_idx]))

        savepath = os.path.join(img_savepath, str(ts_ns)+".png")
        I = cv2.resize(I, (0,0), fx=1./resize_factor, fy=1./resize_factor)
        cv2.imwrite(savepath, I)
        print("Extracted and saved to {}".format(savepath))
        img_idx += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("OpenCameraCalibrator - Image Extractor")
    # Cast the input to string, int or float type 
    parser.add_argument('--path_to_videofile', 
                        default='')
    parser.add_argument('--path_to_image_output', 
                        default='')
    parser.add_argument('--path_to_telemetry', 
                        default='')
    parser.add_argument('--resize_factor', default=1.0, type=float)
    args = parser.parse_args()

    extract_images_from_video_to_folder(args.path_to_videofile, 
        args.path_to_image_output, args.resize_factor, args.path_to_telemetry)