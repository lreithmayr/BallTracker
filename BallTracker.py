import pathlib
import os
import cv2
import sys
import numpy as np
from pykalman import KalmanFilter

current_dir = pathlib.Path(__file__).parent.absolute()
vid = os.path.join(current_dir, "shot2.mp4")
pts = []

def track_roi(tracker, frame, init_bb, pts, frame_nr):
    if frame is None:
        sys.exit()
    if init_bb is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            x_center = int(x + (w / 2))
            y_center = int(y + (h / 2))
            cv2.circle(frame, (x_center, y_center), 0, (0, 255, 0), thickness=2)
            pt = (x_center, y_center)
            pts.append(pt)
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)
        else:
            cv2.putText(frame, "Tracking Failed", (200, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=3)
    cv2.imshow("Frame", frame)
    if frame_nr == 1:
        init_bb = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, init_bb)
    if cv2.waitKey(1) == 27:
        sys.exit()
    return tracker, init_bb

# def trajectory_predictor():




if __name__ == "__main__":
    cap = cv2.VideoCapture(vid)
    timestamps = []
    tracker = cv2.TrackerCSRT_create()
    init_bb = None

    while True:
        check, frame = cap.read()
        t = np.around((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000), 2)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_nr =int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print([t, fps, frame_nr])
        tracker, init_bb = track_roi(tracker, frame, init_bb, pts, frame_nr)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
