from collections import deque
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt


class Tracker(object):

    def __init__(self, cap):
        self.cap = cap
        self.position = deque(maxlen=128)
        self.roi_tracker = cv2.TrackerCSRT_create()
        self.init_bb = None
        self.center = None

    def track_contours(self, cam_frame):
        lt = np.array([65, 33, 71])
        ut = np.array([90, 176, 255])

        blurred = cv2.GaussianBlur(cam_frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lt, ut)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = imutils.grab_contours(cntrs)
        if len(cntrs) > 0:
            c = max(cntrs, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            moments = cv2.moments(c)
            self.center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            self.position.append(self.center)

            if radius > 10:
                cv2.circle(cam_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(cam_frame, self.center, 5, (0, 0, 255), -1)

        return cam_frame, self.center

    def track_roi(self, cam_frame):
        (success, box) = self.roi_tracker.update(cam_frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(cam_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x_center = int(x + (w / 2))
            y_center = int(y + (h / 2))
            cv2.circle(cam_frame, (x_center, y_center), 1, (0, 255, 0), thickness=2)
            cv2.circle(cam_frame, (x_center, y_center), 0, (0, 255, 0), thickness=2)
            pos = (x_center, y_center)
            self.position.append(pos)
            return pos

    def set_init_bb(self, init_bb):
        self.init_bb = init_bb

    def init_roi_tracker(self, cam_frame):
        self.roi_tracker.init(cam_frame, self.init_bb)

    def plot_track(self):
        for pos in self.position:
            y_neg = np.negative(pos[1])
            plt.scatter(pos[0], y_neg)
        plt.show()
