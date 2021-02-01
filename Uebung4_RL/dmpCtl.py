# DMP-based controller.
#
# DMPPARAMS is the struct containing all the parameters of the DMP.
#
# PSI is the vector of basis functions.
#
# Q and QD are the current position and velocity, respectively.

import numpy as np


def dmpCtl(dmpParams, psi_i, q, qd):
    goal = np.array(dmpParams.goal).reshape(-1)
    fw = psi_i.T @ dmpParams.w
    qdd = dmpParams.tau**2 * (dmpParams.alpha * (dmpParams.beta * (goal - q)
                                                 - qd / dmpParams.tau) + fw)

    return qdd

