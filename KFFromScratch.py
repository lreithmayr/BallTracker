import numpy as np

class KFFromScratch(object):

    def __init__(self, dt):
        # Initial state vector [x, y, v_x, v_y]
        self.x_pre = np.array([0, 0, 0, 0]).reshape(4, 1)

        # Transition Matrix F
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]).reshape(4, 4)

        # Control Matrix B
        self.B = np.array([[0.5 * (dt**2), 0], [0, 0.5 * (dt**2)], [dt, 0], [0, dt]]).reshape(4, 2)

        # Measurement Matrix H
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).reshape(2, 4)

        # A priori estimate covariance P
        self.P = (np.diag([1000, 1000, 1000, 1000]).reshape(4, 4)) ** 2

        # Process noise covariance Q
        self.Q = 1e1**2 * np.eye(4)

        # Control vector u: Acceleration
        self.u = np.array([0, -9.81]).reshape(2, 1)

        # Observation/Measurement noise covariance R
        self.R = 1e-3 * np.eye(2)

    def predict(self):
        # Predicted a-priori state estimate
        x_k = np.matmul(self.F, self.x_pre) + np.matmul(self.B, self.u)

        # Predicted a-priori estimate covariance
        p_k = np.matmul(np.matmul(self.F, self.P), np.transpose(self.F)) + self.Q

        # Measurement
        z_k = np.matmul(self.H, x_k)

        return x_k, p_k, z_k

    def update(self, x_k, p_k, z_k, z):
        # Pre-fit measurement residual y_wave
        y_k = z - z_k

        # Innovation covariance S
        s_k = np.matmul(np.matmul(self.H, p_k), np.transpose(self.H)) * self.R

        # Optimal Kalman gain K
        k_k = np.matmul(np.matmul(p_k, np.transpose(self.H)), np.invert(s_k))

        # Updated state estimate x_k_updt
        x_k_updt = x_k + np.matmul(k_k, y_k)

        # Updated estimate covariance
        p_k_updt = np.matmul((np.eye(len(self.P)) - np.matmul(k_k, self.H)), p_k)

        return x_k_updt, p_k_updt