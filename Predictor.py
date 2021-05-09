from pykalman import KalmanFilter
from Tracker import Tracker

class Predictor(object):
    # Define state vector for Karman filter
    x_k = (x, x_dot, y, y_dot)