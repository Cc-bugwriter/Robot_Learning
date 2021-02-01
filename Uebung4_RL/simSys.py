from getDMPBasis import *
from dmpCtl import *


def simSys(states, dmpParams, dt, nSteps):

    Phi = getDMPBasis(dmpParams, dt, nSteps)

    for i in range(nSteps - 1):
        qdd = dmpCtl(dmpParams, Phi[i, :].transpose(), states[i, ::2], states[i, 1::2].transpose())
        #raw_input()
        states[i + 1, 1::2] = states[i, 1::2] + dt * qdd
        states[i + 1, ::2] = states[i, ::2] + dt * states[i, 1::2]

    return states
