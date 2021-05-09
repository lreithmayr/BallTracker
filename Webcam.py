import cv2
import imutils
import numpy as np
from pykalman import KalmanFilter
from Tracker import Tracker
import matplotlib.pyplot as plt

if __name__ == "__main__":
    vid = "http://192.168.0.94:8080/video"
    cap = cv2.VideoCapture(vid)
    tracker = Tracker(cap)

    transition_matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
    observation_matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]
    init_cov = 1.0e-3 * np.eye(4)
    transition_cov = 1.0e-4 * np.eye(4)
    observation_cov = 1.0e-1 * np.eye(2)

    while True:
        check, frame = cap.read()
        frame = imutils.resize(frame, int(500), int(500))
        tracked_frame = tracker.track_contours(frame)
        pos = tracker.get_position()
        vel = tracker.get_velocity()
        cv2.imshow("Frame", tracked_frame)
        if len(pos) > 1:
            x_init = pos[-1][0]
            print(x_init)
            y_init = pos[-1][1]
            print(y_init)
            v_x_init = vel[-1][0]
            v_y_init = vel[-1][1]
            init_state = [x_init, y_init, v_x_init, v_y_init]
            kf = KalmanFilter(transition_matrices=transition_matrix, observation_matrices=observation_matrix,
                              transition_covariance=transition_cov,
                              observation_covariance=observation_cov,
                              initial_state_mean=init_state, initial_state_covariance=init_cov)
            (filtered_state_means, filtered_state_covariances) = kf.filter(pos)
            plt.plot(pos[-1][0], pos[-1][1], 'xr', label='measured')
            plt.axis([0, 520, 360, 0])
            plt.plot(filtered_state_means[:, 0], filtered_state_means[:, 1], 'ob', label='kalman output')
            plt.legend(loc=2)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.show()
    tracker.plot_track()