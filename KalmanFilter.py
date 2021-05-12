import cv2
import numpy as np


class KalmanFilter(object):

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = 3e5 * np.eye(4).astype(np.float32)
        self.kf.measurementNoiseCov = 1e-3 * np.eye(2).astype(np.float32)
        self.kf.errorCovPost = np.diag([10, 10, 10, 10])**2

    def estimate_position(self, pos_x, pos_y):
        measured = np.array([np.float32(pos_x), np.float32(pos_y)])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

    def get_errorCovPos(self):
        print(self.kf.errorCovPost)
        return self.kf.errorCovPost
