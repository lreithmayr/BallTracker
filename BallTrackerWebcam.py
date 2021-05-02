import cv2
import numpy as np
# from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import imutils
from collections import deque


def track_contours(frame, pts):

    lt = np.array([65, 33, 71])
    ut = np.array([90, 176, 255])

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lt, ut)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cntrs = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = imutils.grab_contours(cntrs)
    center = None

    if len(cntrs) > 0:
        print("Works")
        c = max(cntrs, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    else:
        print ("Fail")

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)

    return frame, pts


def plot_track(pts):
    for pt in pts:
        y_neg = np.negative(pt[1])
        plt.scatter(pt[0], y_neg)
    plt.show()


if __name__ == "__main__":
    vid = "http://192.168.0.94:8080/video"
    pts = deque(maxlen=64)

    cap = cv2.VideoCapture(vid)

    while True:
        check, frame = cap.read()
        frame = imutils.resize(frame, int(1920/1.5), int(1080/1.5))
        frame, pts = track_contours(frame, pts)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    print(pts)

    plot_track(pts)