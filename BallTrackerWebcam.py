import cv2
import numpy as np
# from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import imutils


def contour_tracker(tracker, frame, init_bb, pts):
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
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness=1)
        else:
            cv2.putText(frame, "Tracking Failed", (200, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=3)

    if cv2.waitKey(1) == ord("s"):
        init_bb = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, init_bb)
    return tracker, init_bb, pts


def plot_track(pts):
    for pt in pts:
        y_neg = np.negative(pt[1])
        plt.scatter(pt[0], y_neg)
    plt.show()


if __name__ == "__main__":
    vid = "http://192.168.0.94:8080/video"
    pts = []

    cap = cv2.VideoCapture(vid)

    while True:
        check, frame = cap.read()
        frame = imutils.resize(frame, 1000, 1000)
        cv2.imshow("Frame", frame)
        tracker, init_bb, pts = contour_tracker(tracker, frame, init_bb, pts)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    plot_track(pts)