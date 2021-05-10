import cv2
import numpy as np


class KalmanFilter(object):

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = 1e-5
        self.kf.measurementNoiseCov =

    def estimate_position(self, pos_x, pos_y):
        measured = np.array([[np.float32(pos_x)], [np.float32(pos_y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted
