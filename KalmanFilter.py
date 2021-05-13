import cv2
import numpy as np
import inspect


class KalmanFilter(object):

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = 3e5 * np.eye(4).astype(np.float32)
        self.kf.measurementNoiseCov = 1e-3 * np.eye(2).astype(np.float32)
        self.kf.errorCovPost = np.diag([1, 1, 1, 1])**2
        self.caller_name = inspect.stack()[1][0].f_code.co_filename
        self.predicted = None

    def estimate_position(self, pos_x, pos_y, cam_frame, radius=None):
        measured = np.array([[np.float32(pos_x)], [np.float32(pos_y)]])
        self.kf.correct(measured)
        self.predicted = self.kf.predict()
        if "Video" in self.caller_name:
            cv2.circle(cam_frame, (self.predicted[0], self.predicted[1]), 15, [0, 45, 255], 2, 8)
        elif "Webcam" in self.caller_name and radius is not None:
            cv2.circle(cam_frame, (self.predicted[0], self.predicted[1]), int(radius), [0, 45, 255], 2, 8)
        return self.predicted

    def get_error_cov_pos(self):
        return self.kf.errorCovPost

    def draw_estimation_error(self, frame):
        err_x = 2 * np.sqrt(self.get_error_cov_pos()[0][0])
        err_y = 2 * np.sqrt(self.get_error_cov_pos()[1][1])
        uncert = (err_x + err_y) / 2
        cv2.circle(frame, (self.predicted[0], self.predicted[1]), int(uncert), [0, 45, 255], 2, 8)
