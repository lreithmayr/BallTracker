import cv2
import numpy as np
import inspect


class KalmanFilter(object):

    def __init__(self, dt):
        # Initialize KF with 4 state variables [pos_x(k), pos_y(k), v_x(k), v_y(k)]
        # and two measurement variables [(pos_x(k), pos_y(k))]
        self.kf = cv2.KalmanFilter(4, 2)

        # Transition Matrix F
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]).reshape(
            4, 4).astype(np.float32)

        # Control Matrix B
        self.kf.controlMatrix = np.array([[0.5 * (dt ** 2), 0], [0, 0.5 * (dt ** 2)], [dt, 0], [0, dt]]) \
            .reshape(4, 2).astype(np.float32)

        # Measurement Matrix H
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).reshape(2, 4).astype(np.float32)

        # Process Noise Covariance Q
        self.kf.processNoiseCov = 1e1**2 * np.eye(4).astype(np.float32)

        # Observation Noise Covariance R
        self.kf.measurementNoiseCov = 1e-3 * np.eye(2).astype(np.float32)

        # Initial a-priori Erro Covariance Matrix P
        self.kf.errorCovPre = (np.diag([1000, 1000, 1000, 1000]).astype(np.float32))**2

        # Gets name of calling file
        self.caller_name = inspect.stack()[1][0].f_code.co_filename

        self.predicted = None

    def estimate_position(self, pos_x, pos_y, cam_frame, radius=None):
        measured = np.array([[np.float32(pos_x)], [np.float32(pos_y)]])
        self.kf.correct(measured)
        self.predicted = self.kf.predict()

        # Handling call by Video or Webcam
        if "Video" in self.caller_name:
            cv2.circle(cam_frame, (self.predicted[0], self.predicted[1]), 15, [0, 20, 255], 2, 8)
        elif "Webcam" in self.caller_name and radius is not None:
            cv2.circle(cam_frame, (self.predicted[0], self.predicted[1]), int(radius), [0, 45, 255], 2, 8)
        return self.predicted

    def get_error_cov_pre(self):
        return self.kf.errorCovPre

    def draw_estimation_error(self, frame):
        err_x = 2 * np.sqrt(self.get_error_cov_pre()[0][0])
        err_y = 2 * np.sqrt(self.get_error_cov_pre()[1][1])
        uncert = (err_x + err_y) / 2
        cv2.circle(frame, (self.predicted[0], self.predicted[1]), int(uncert), [0, 45, 255], 2, 8)
