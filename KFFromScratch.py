import numpy as np

class KFFromScratch(object):

    """"
    Nomenclature from Wikipedia article on Kalman filters.

    State vector:

        x(k) = [pos_x(k)
                pos_y(k)
                vel_x(k)
                vel_y(k)]

    Transition model:

        pos_x(k) = pos_x(k-1) + (vel_x(k-1) * dt) + 0.5 * dt^2 * a_x(k-1)
        pos_y(k) = pos_y(k-1) + (vel_y(k-1) * dt) + 0.5 * dt^2 * a_y(k)
        vel_x(k) = vel_x(k-1) + dt * a_x(k)
        vel_y(k) = vel_y(k-1) + dt * a_y(k)

    Gives:

               [1 0 dt 0      [pos_x(k-1)        [0.5 * dt^2    0
                0 1 0 dt       pos_y(k-1)          0        0.5 * dt^2          [a_x(k)
        x(k) =  0 0 1 0    *   vel_x(k-1)    +     dt           0         *      a_y(k)]
                0 0 0 1]       vel_y(k-1)]         0           dt]


             =    F        *      x(k-1)     +           B                *         a(k)


    """

    def __init__(self):
        self.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
