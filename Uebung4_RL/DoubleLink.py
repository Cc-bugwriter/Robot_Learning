from math import pi, sin, cos
import numpy as np
from matplotlib import pyplot as plt


class DoubleLink():
    def __init__(self):

        self.PDGains = []
        self.PDSetPoints = []

        self.lengths = np.array([1.0, 1.0])
        self.masses = np.array([1.0, 1.0])
        self.friction = np.array([0.025, 0.025])
        self.minRangeState = np.array([-0.8, -50.0, -2.5, -50.0])
        self.maxRangeState = np.array([0.8, 50.0, .05, 50.0])
        self.minRangeAction = np.array([-20.0, -20.0])
        self.maxRangeAction = np.array([20.0, 20.0])
        self.inertias = self.masses * (self.lengths ** 2 + 0.0001) / 12.0
        self.g = 9.81
        self.sim_dt = 1e-4  # 2e-3;
        self.dimJoints = 2
        self.dimState = 4

        self.sampleInitStateFunc = 0

    def getDynamicsMatrices(self, state):
        m1 = self.masses[0]
        m2 = self.masses[1]

        l1 = self.lengths[0]
        l2 = self.lengths[1]

        lg1 = l1 / 2.0
        lg2 = l2 / 2.0

        q1 = state[0] + pi / 2
        q2 = state[2]

        q1d = state[1]
        q2d = state[3]

        s1 = sin(q1)
        c1 = cos(q1)

        s2 = sin(q2)
        c2 = cos(q2)

        s12 = sin(q1 + q2)
        c12 = cos(q1 + q2)

        M = np.mat(
            [[m1 * lg1 ** 2 + self.inertias[0] + m2 * (l1 ** 2 + lg2 ** 2 + 2 * l1 * lg2 * c2) + self.inertias[1], m2 *
              (lg2 ** 2 + l1 * lg2 * c2) + self.inertias[1]], [m2 * (lg2 ** 2 + l1 * lg2 * c2) + self.inertias[1], m2 *
                                                               lg2 ** 2 + self.inertias[1]]])

        gravity = [m1 * self.g * lg1 * c1 + m2 * self.g * (l1 * c1 + lg2 * c12), m2 * self.g * lg2 * c12]

        coriolis = [2 * m2 * l1 * lg2 * (q1d * q2d * s2 + q1d ** 2 * s2) + self.friction[0] * q1d, 2 * m2 * l1 * lg2 *
                    (q1d ** 2 * s2) + self.friction[1] * q2d]

        return gravity, coriolis, M

    # Get the jacobian at a given angle position q for a given link numLink
    def getJacobian(self, theta, numLink):

        if 'numLink' not in locals():
            numLink = self.dimJoints

        si = self.getForwardKinematics(theta, numLink);

        J = np.zeros((2, self.dimJoints))

        lengths = self.lengths

        for j in range(numLink):
            pj = np.array([0.0, 0.0])
            for i in range(j):
                pj = pj + [sin(sum(theta[:, :(i + 1)])), cos(sum(theta[:, :(i + 1)]))] * lengths[i]

            pj = -(si - pj)
            J[np.ix_([0, 1], [j])] = np.mat([-pj[1], pj[0]]).transpose()

        return J, si

    def getForwardKinematics(self, theta, numLink):

        if 'numLink' not in locals():
            numLink = self.dimJoints

        lengths = self.lengths
        y = np.array(np.zeros((theta.shape[0], 2)))[0]
        for i in range(numLink):
            y += np.array([sin(np.sum(theta[:, :i + 1], 1)), cos(np.sum(theta[:, :i + 1], 1))]) * lengths[i]

        return y

    def getJointsInTaskSpace(self, q):

        x1 = np.array(self.lengths[0] * np.array([sin(q[0]), cos(q[0])]))
        x2 = x1 + np.array(self.lengths[1] * np.array([sin(q[2] + q[0]), cos(q[2] + q[0])]))

        return x1, x2

    def visualize(self, q, line):
        lw = 4.0
        fs = 26

        mp1, mp2 = self.getJointsInTaskSpace(q)

        # display the link in red if the bounds are not good, blue if they are good

        thisx = [0, mp1[0], mp2[0]]
        thisy = [0, mp1[1], mp2[1]]

        line.set_data(thisx, thisy)
