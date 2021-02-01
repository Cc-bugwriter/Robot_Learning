#Generates the trajectory we want to imitate.
import numpy as np
import scipy.interpolate as interp
from math import pi


def getImitationData(dt, time, multiple_demos=False):

    if multiple_demos:
        np.random.seed(10)
        nSteps = len(time)
        nDemo = 45
        r = np.random.randn(nDemo, 1) * 0.05 + 2
        qs = np.sin(2 * pi * np.outer(r, time[500:1001]))

        x = np.insert(time[500:1001], 0, np.array([0, dt]))
        x = np.append(x, np.array([time[-2], time[-1]]))

        q = np.zeros((nDemo, nSteps))
        for i in range(nDemo):
            y = np.insert(qs[i, :], 0, np.array([-pi, -pi]))
            y = np.append(y, np.array([0.3, 0.3]))
            f = interp.InterpolatedUnivariateSpline(x, y)
            q[i, :] = f(time) + np.random.randn(1) * 0.35 + 1

    else:
        qs = np.sin(2 * pi * 2 * time[500:1001])
        x = np.insert(time[500:1001], 0, np.array([0, dt]))
        x = np.append(x, np.array([time[-2], time[-1]]))
        y = np.insert(qs, 0, np.array([-pi, -pi]))
        y = np.append(y, np.array([0.3, 0.3]))

        f1 = interp.InterpolatedUnivariateSpline(x, y)
        q1 = f1(time)

        f2 = interp.InterpolatedUnivariateSpline(np.array([0, dt, time[-2], time[-1]]), np.array([0, 0, -0.8, -0.8]))
        q2 = f2(time)

        q = np.vstack((q1, q2))

    qd = np.diff(q) / dt
    qd = np.hstack((qd, qd[:, -1, None]))
    qdd = np.diff(qd) / dt
    qdd = np.hstack((qdd, qdd[:, -1, None]))

    return [q, qd, qdd]
