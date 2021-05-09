from pykalman import KalmanFilter
from Tracker import Tracker
import numpy as np

class Predictor(object):

    def __init__(self):
        self.transition_matrix = [[1, 0, 1, 0], [0, 1, 0, 1]]
        self.observation_matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]

    def filter(self):


        kf = KalmanFilter(self.transition_matrix, self.observation_matrix,)
