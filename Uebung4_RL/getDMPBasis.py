# Generates a vector of uniformly distributed radial basis functions.
# Returns the normalized basis times the phase
import numpy as np


def getDMPBasis (params, dt, nSteps):

    nBasis = params.nBasis
    Ts = nSteps * dt - dt

    C = np.zeros(nBasis)  # Basis function centres
    H = np.zeros(nBasis)  # Basis function bandwidths

    for i in range(nBasis):
        C[i] = np.exp(-params.alphaz * (i) / (nBasis-1) * Ts)

    for i in range(nBasis - 1):
        H[i] = 0.5 / (0.65 * (C[i+1] - C[i])**2)

    H[nBasis-1] = H[nBasis-2]

    X = 1 # Canonical system
    Phi = np.zeros((nSteps, nBasis))

    for k in range(nSteps):
        for j in range(nBasis):
            Phi[k, j] = np.exp(-H[j] * (X - C[j]) ** 2)  # Basis function activation over time
        Phi[k, :] = (Phi[k, :] * X) / np.sum(Phi[k, :])  # Normalize basis functions and weight by canonical state
        X = X - params.alphaz * X * params.tau * dt

    return Phi
