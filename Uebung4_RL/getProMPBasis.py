import numpy as np
import matplotlib.pyplot as plt

def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):

    time = np.arange(dt, nSteps*dt, dt)
    Ts = time[-1]

    center_time = np.linspace(-2*bandwidth, Ts + 2*bandwidth, n_of_basis)

    Phi = np.zeros((nSteps, n_of_basis))

    for k in range(nSteps):
        for j in range(n_of_basis):
            # Basis function activation over time
            Phi[k, j] = np.exp(-0.5 * (time[k] - center_time[j]) ** 2 / bandwidth**2)
        # Normalize basis functions and weight by canonical state
        Phi[k, :] = Phi[k, :] / np.sum(Phi[k, :])

    # manuel plot or not
    plotFigure = False

    if plotFigure:
        plt.figure()
        for i in range(n_of_basis):
            plt.plot(time, Phi[:, i], label='basis function at ' + '%.2f' % center_time[i])
        plt.title("Radial Basis Function")
        plt.legend(bbox_to_anchor=(0.97, 0.5), loc='center left')
        plt.show()

    return Phi
