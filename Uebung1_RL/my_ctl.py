# CTL is the name of the controller.
# Q_HISTORY is a matrix containing all the past position of the robot. Each row of this matrix is [q_1, ... q_i], where
# i is the number of the joints.
# Q and QD are the current position and velocity, respectively.
# Q_DES, QD_DES, QDD_DES are the desired position, velocity and acceleration, respectively.
# GRAVITY is the gravity vector g(q).
# CORIOLIS is the Coriolis force vector c(q, qd).
# M is the mass matrix M(q).

import numpy as np

def my_ctl(ctl, q, qd, q_des, qd_des, qdd_des, q_hist, q_deshist, gravity, coriolis, M, Gain=1):
    KP = Gain * np.diag([60, 30])
    KD = Gain * np.diag([10, 6])
    KI = Gain * np.diag([0.1, 0.1])
    if ctl == 'P':
        u = KP @ (q_des - q)
    elif ctl == 'PD':
        u = KP @ (q_des - q) + KD @ (qd_des - qd)
    elif ctl == 'PID':
        u = KP @ (q_des - q) + KD @ (qd_des - qd) + \
            KI @ (np.sum(q_deshist, axis=0) - np.sum(q_hist, axis=0))
    elif ctl == 'PD_Grav':
        u = KP @ (q_des - q) + KD @ (qd_des - qd) + gravity
    elif ctl == 'ModelBased':
        qdd_ref = qdd_des + KP @ (q_des - q) + KD @ (qd_des - qd)
        u = M @ qdd_ref + coriolis + gravity
    return u.reshape(-1, 1)
