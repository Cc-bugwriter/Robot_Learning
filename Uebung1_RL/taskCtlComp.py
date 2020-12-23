# This is one of the two main classes you need to run.
#
# CTLS is a list of cells with your controller name
# ('JacTrans','JacPseudo,'JacDPseudo','JacNullSpace'). You can run more controllers
# one after another, e.g. passing {'JacTrans','JacPseudo'}.
#
# PAUSETIME is the number of seconds between each iteration for the
# animated plot. If 0 only the final position of the robot will be
# displayed.

import numpy as np
from simSys import *
from DoubleLink import *


def taskCtlComp(ctls=['JacDPseudo'], pauseTime=False, resting_pos=None):
    dt = 0.002
    robot = DoubleLink()
    robot.friction = np.array([2.5, 2.5])
    t_end = 3.0
    time = np.arange(0, t_end, dt)
    nSteps = len(time)
    numContrlComp = len(ctls)
    target = {}
    target['x'] = np.tile([-0.35, 1.5], (nSteps, 1))
    target['cartCtl'] = True
    states = simSys(robot, dt, nSteps, ctls, target, pauseTime, resting_pos)
