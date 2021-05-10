import pathlib
import cv2
import os
from Tracker import Tracker
from KalmanFilter import KalmanFilter
import imutils

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    vid = os.path.join(current_dir, "shot2.mp4")

    cap = cv2.VideoCapture(vid)
    tracker = Tracker(cap)
    kf = KalmanFilter()

    while True:
        check, frame = cap.read()
        if frame is None:
            break
        frame = imutils.resize(frame, 1000, 500)
        frame_nr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if tracker.init_bb is not None:
            frame, pos = tracker.track_roi(frame)
            # print(pos, pos[0], pos[1])
            pred_pos = kf.estimate_position(pos[0], pos[1])
            cv2.circle(frame, (pred_pos[0], pred_pos[1]), 20, [0,255,255], 2, 8)
            cv2.imshow("Frame", frame)
        if frame_nr == 1:
            cv2.imshow("Frame", frame) 
            tracker.set_init_bb(cv2.selectROI("Frame", frame, fromCenter=False,
                                              showCrosshair=True))
            tracker.init_roi_tracker(frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.plot_track()
