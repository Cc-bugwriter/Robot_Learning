# Launches the simulation for the DMP-based controller and plot the
# results.
#
# GOALS is an optional array of cells that specifies the desided positions
# and velocities of the joints.
#
# GOALS is an optional array of values of different taus for the DMP.
#
# FILENAME is the name of your output files. In the end the code will
# generate two pdf files named 'filename_q1.pdf' and
# 'filename_q2.pdf' containing the plots.

import numpy as np
from simSys import *
from DoubleLink import *
from getImitationData import *
from dmpTrain import *
from math import pi


def dmpComparison(goals, taus, filename):

    dt = 0.002

    robot = DoubleLink()
    robot.friction = np.array([2.5, 2.5])

    t_end = 3.0

    sim_time = np.arange(dt, t_end-dt, dt)
    nSteps = len(sim_time)

    data = getImitationData(dt, sim_time)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    dmpParams = dmpTrain(q, qd, qdd, dt, len(sim_time))

    states = np.zeros((nSteps, 4))
    states[0, ::2] = [-pi, 0]
    states = simSys(states, dmpParams, dt, nSteps)

    f1 = plt.figure()
    plt.plot(sim_time, q[0, :], linewidth=2.0, label='Desired $q_1$')
    plt.plot(sim_time, states[:, 0], ':', color='r', linewidth=4.0, label='DMP $q_1$')

    f2 = plt.figure()
    plt.plot(sim_time, q[1, :], linewidth=2.0, label='Desired $q_2$')
    plt.plot(sim_time, states[:, 2], ':', color='r', linewidth=4.0, label='DMP $q_2$')

    dmpParamsOld = dmpParams

    p1_h = [0, 0]
    p2_h = [0, 0]

    if goals != []:
        for i in range(len(goals)):
            states = np.zeros((nSteps, 4))
            states[0, ::2] = [-pi, 0]
            dmpParams.goal = goals[i]
            states = simSys(states, dmpParams, dt, nSteps)

            plt.figure(f1.number)
            plt.plot(sim_time, states[:, 0], linewidth=2.0, label='DMP $q_1$ with goal = [' + str(goals[i][0]) + ']')
            plt.plot(sim_time[-1], goals[i][0], 'kx', markersize=15.0)

            plt.figure(f2.number)
            plt.plot(sim_time, states[:,2], linewidth=2.0, label='DMP $q_2$ with goal = [' + str(goals[i][1]) + ']')
            plt.plot(sim_time[-1], goals[i][1], 'kx', markersize=15.0)


    dmpParams = dmpParamsOld

    if taus != []:
        for i in range(len(taus)):
            states = np.zeros((nSteps, 4))
            states[0,::2] = [-pi, 0]
            dmpParams.tau = taus[i]
            states = simSys ( states, dmpParams, dt, nSteps )

            plt.figure(f1.number)
            plt.plot(sim_time, states[:,0], linewidth=2.0, label=r'DMP $q_1$ with $\tau$ = [' + str(taus[i]) + ']')

            plt.figure(f2.number)
            plt.plot(sim_time, states[:,2], linewidth=2.0, label=r'DMP $q_2$ with $\tau$ = [' + str(taus[i]) + ']')


    plt.figure(f1.number)
    plt.legend(loc=0)
    plt.figure(f2.number)
    plt.legend(loc=0)

    plt.draw_all()
    plt.pause(0.001)


if __name__ == "__main__":
    dt = 0.002

    robot = DoubleLink()
    robot.friction = np.array([2.5, 2.5])
    t_end = 3.0

    sim_time = np.arange(dt, t_end - dt, dt)
    nSteps = len(sim_time)

    data = getImitationData(dt, sim_time)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    dmpParams = dmpTrain(q, qd, qdd, dt, len(sim_time))
    Phi = getDMPBasis(dmpParams, dt, nSteps)

    states = np.zeros((nSteps, 4))
    states[0, ::2] = [-pi, 0]

    states = simSys(states, dmpParams, dt, nSteps)

    tau = 1
    beta = 6.25
    alpha = 25
    goal = q[-1]

    # Compute the forcing function
    ft = qdd / tau ** 2 - alpha * (beta * (goal - q) - qd / tau)

    # Learn the weights
    w = np.linalg.pinv(Phi.T @ Phi + 0.01 * np.ones((Phi.T @ Phi).shape)) @ Phi.T @ ft.T