from collections import deque
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt


class Tracker(object):

    def __init__(self, cap):
        self.cap = cap
        self.position = deque(maxlen=128)
        self.velocity = deque(maxlen=128)
        self.velocity.append(None)
        self.accel = deque(maxlen=128)
        self.accel.append(None)
        self.roi_tracker = cv2.TrackerCSRT_create()
        self.init_bb = None

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
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            self.position.append(center)

            if radius > 10:
                cv2.circle(cam_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(cam_frame, center, 5, (0, 0, 255), -1)

        # for i in range(1, len(self.position)):
        #     if self.position[i - 1] is None or self.position[i] is None:
        #         continue
        #     thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        #     cv2.line(cam_frame, self.position[i - 1], self.position[i], (0, 255, 0), thickness)
        return cam_frame

    def set_init_bb(self, init_bb):
        self.init_bb = init_bb

    def init_roi_tracker(self, cam_frame):
        self.roi_tracker.init(cam_frame, self.init_bb)

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
            return cam_frame

    def plot_track(self):
        for pos in self.position:
            y_neg = np.negative(pos[1])
            plt.scatter(pos[0], y_neg)
        plt.show()

    def get_position(self):
        return self.position

    def compute_velocity(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        dt = 1 / fps
        if len(self.get_position()) > 1:
            dx = self.position[-1][0] - self.position[-2][0]
            dy = self.position[-1][1] - self.position[-2][1]
            v_x = dx/dt
            v_y = dy/dt
            self.velocity.append((v_x, v_y))

    def get_velocity(self):
        self.compute_velocity()
        return self.velocity

    def compute_accel(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        dt = 1 / fps
        vel = self.velocity
        if len(vel) > 2:
            acc_x = (vel[-1][0] - vel[-2][0]) / dt
            acc_y = (vel[-1][1] - vel[-2][1]) / dt
            self.accel.append((acc_x, acc_y))

    def get_accel(self):
        self.compute_accel()
        return self.accel