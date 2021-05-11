import pathlib
import cv2
import os
from Tracker import Tracker
from KalmanFilter import KalmanFilter
import imutils
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    vid = os.path.join(current_dir, "shot2.mp4")

    cap = cv2.VideoCapture(vid)
    tracker = Tracker(cap)
    kf = KalmanFilter()
    predictions = []

    while True:
        check, frame = cap.read()
        if not check:
            break
        frame = imutils.resize(frame, 1000, 500)
        frame_nr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cv2.imshow("Frame", frame)

        if tracker.init_bb is not None:
            pos = tracker.track_roi(frame)
            pred_pos = kf.estimate_position(pos[0], pos[1])
            cv2.circle(frame, (pred_pos[0], pred_pos[1]), 20, [0, 255, 255], 2, 8)
            predictions.append((pred_pos[0], pred_pos[1]))
            cv2.imshow("Frame", frame)
        if frame_nr == 1:
            tracker.set_init_bb(cv2.selectROI("Frame", frame, fromCenter=False,
                                              showCrosshair=True))
            tracker.init_roi_tracker(frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(predictions[5][0])

    for (pos, pred) in zip(tracker.get_position(), predictions):
        y_neg = np.negative(pos[1])
        y_neg_pred = np.negative(pred[1])
        plt.scatter(pos[0], y_neg)
        plt.scatter(pred[0], y_neg_pred)
    plt.show()