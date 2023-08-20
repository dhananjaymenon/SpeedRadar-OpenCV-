import cv2
import os
import errno
import sys
from tracker import EuclideanDistTracker, start_line, end_line
import numpy as np


"""
Region of interest 
 - determined manually through pixels
 - To be changed for a different video
 
roi = frame[height_min: height_max, width_min: width_max]
"""
height_min = 50
height_max = 540
width_min = 200
width_max = 960

"""
Create Tracker Object
"""
tracker = EuclideanDistTracker()

# Create background subtractor model
object_detector = cv2.createBackgroundSubtractorMOG2(history=None, varThreshold=None)

"""
KERNELS 
 - Results may vary depending on the clarity of the video
 - Alter these values to fine tune vehicle detection
 - Displaying (or imshow'ing) the mask or erode frame will help
   in determining correct values 
 
"""
kernelOp = np.ones((3, 3), np.uint8)
kernelOp2 = np.ones((5, 5), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernel_e = np.ones((5, 5), np.uint8)


def mask_method_1(roi: np.ndarray) -> np.ndarray:
    """
    takes a frame, converts it to binary and alters pixels for better detection
    The below pixel values have been chosen by trial and error for a particular video
    """
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    return mask


def mask_method_2(roi: np.ndarray) -> np.ndarray:
    """
    takes a frame, converts it to binary and alters pixels for better detection
    The below pixel values have been chosen by trial and error for a particular video
    """
    fgmask = fgbg.apply(roi)
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernelCl)
    return mask2


def draw_lines_on_frame(roi: np.ndarray):
    """
    Lines are drawn as per the start and end lines described in tracker.py
    """
    color_red = (0, 0, 255)
    line_width = 2
    cv2.line(roi, (0, start_line), (960, start_line), color_red, line_width)
    cv2.line(roi, (0, start_line + 20), (960, start_line + 20), color_red, line_width)

    cv2.line(roi, (0, end_line), (960, end_line), color_red, line_width)
    cv2.line(roi, (0, end_line + 20), (960, end_line + 20), color_red, line_width)


def run_speed_tracker(video_file_location: str):
    if not os.path.exists(video_file_location):
        raise FileNotFoundError(errno.ENOENT, "Video file location is incorrect", video_file_location)
        sys.exit(1)

    cap = cv2.VideoCapture(video_file_location)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / (fps - 1))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        height, width, _ = frame.shape
        # print(height,width)
        # 540,960

        # Extract ROI
        roi = frame[height_min: height_max, width_min: width_max]

        # MASKING METHOD 1
        mask = mask_method_1(roi)

        # DIFFERENT MASKING METHOD 2 -> This is used
        mask = mask_method_2(roi)
        e_img = cv2.erode(mask, kernel_e)  # Eroding frame for better detection

        # Creating contours
        contours, _ = cv2.findContours(e_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # THRESHOLD
            if area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                detections.append([x, y, w, h])

        # Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id

            if tracker.getsp(id) < tracker.limit():
                cv2.putText(roi, str(id) + " " + str(tracker.getsp(id)), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 0), 2)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                cv2.putText(roi, str(id) + " " + str(tracker.getsp(id)), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 255), 2)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

            s = tracker.getsp(id)
            if tracker.f[id] == 1 and s != 0:
                tracker.capture(roi, x, y, h, w, s, id)

        # DRAW LINES
        draw_lines_on_frame(roi)

        # DISPLAY
        # cv2.imshow("Mask",mask2)
        # cv2.imshow("Erode", e_img)
        cv2.imshow("ROI", roi)

        key = cv2.waitKey(wait_time - 10)

        """
        Press ESC to exit program
        """
        if key == 27:
            tracker.end()
            # end=1
            break

    cap.release()
    cv2.destroyAllWindows()
