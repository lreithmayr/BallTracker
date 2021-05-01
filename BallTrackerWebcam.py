import cv2
import numpy as np
# from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import imutils


def track_contours(frame, pts):
    lower_thr = (29, 86, 6)
    upper_thr = (64, 255, 255)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_thr, upper_thr)
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

    pts.append(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness=1)

    return pts, frame, mask


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
        frame = imutils.resize(frame, 700, 700)
        pts, frame, mask = track_contours(frame, pts)
        cv2.imshow("Frame", mask)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # plot_track(pts)