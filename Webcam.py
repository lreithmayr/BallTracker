import cv2
import imutils
from KalmanFilter import KalmanFilter
from Tracker import Tracker
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    vid = "http://192.168.0.94:8080/video"
    cap = cv2.VideoCapture(vid)
    tracker = Tracker(cap)
    kf = KalmanFilter()
    predictions = []

    while True:
        check, frame = cap.read()
        frame = imutils.resize(frame, int(500), int(500))
        frame, radius = tracker.track_contours(frame)
        pos = tracker.get_center()
        cv2.imshow("Frame", frame)

        if tracker.center is not None:
            pred_pos = kf.estimate_position(pos[0], pos[1])
            if radius is not None:
                cv2.circle(frame, (pred_pos[0], pred_pos[1]), int(radius), [0, 60, 255], 2, 8)
            predictions.append(pred_pos)
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    plt_act = None
    plt_pred = None
    for (pos, pred) in zip(tracker.get_position(), predictions):
        y_neg = np.negative(pos[1])
        y_neg_pred = np.negative(pred[1])
        plt_act = plt.scatter(pos[0], y_neg, marker=6, c="indianred", label="Actual Position")
        plt_pred = plt.scatter(pred[0], y_neg_pred, marker="x", c="mediumseagreen", label="Predicted Position")

    plt.legend(handles=[plt_act, plt_pred])
    plt.show()