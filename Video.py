import pathlib
import cv2
import numpy as np
import os
from Tracker import Tracker
from KalmanFilter import KalmanFilter
import imutils


def mask_frame(cam_frame):
    lt = np.array([0, 58, 105])
    ut = np.array([179, 255, 255])

    blurred = cv2.GaussianBlur(cam_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lt, ut)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    vid = os.path.join(current_dir, "shot2.mp4")

    cap = cv2.VideoCapture(vid)
    tracker = Tracker(cap)
    kf = KalmanFilter()
    masked_frame = None

    while True:
        check, frame = cap.read()
        if not check:
            break
        frame = imutils.resize(frame, 1000, 500)
        frame_nr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cv2.imshow("Frame", frame)
        masked_frame = mask_frame(frame) 
        if tracker.init_bb is not None:
            pos = tracker.track_roi(masked_frame)
            cv2.circle(frame, pos, 1, (0, 255, 0), thickness=2)
            cv2.circle(frame, pos, 0, (0, 255, 0), thickness=2)
            pred_pos = kf.estimate_position(pos[0], pos[1])
            cv2.circle(frame, (pred_pos[0], pred_pos[1]), 20, [0, 255, 255], 2, 8)
            cv2.imshow("Frame", frame)
        if frame_nr == 1:
            tracker.set_init_bb(cv2.selectROI("Mask", masked_frame, fromCenter=False,
                                              showCrosshair=True))
            tracker.init_roi_tracker(masked_frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.plot_track()
