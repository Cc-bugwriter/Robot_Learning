import numpy as np
import matplotlib.pyplot as plt
from getImitationData import *
from getProMPBasis import *


def proMP(nBasis, condition=False):

    dt = 0.002
    time = np.arange(dt, 3, dt)
    nSteps = len(time)
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    bandwidth = 0.2
    Phi = getProMPBasis(dt, nSteps, nBasis, bandwidth)

    # compute w for each trajectory
    w_list = []
    for q_i in q:
        q_i = q_i.reshape(1, -1)
        w = np.linalg.pinv(Phi.T @ Phi + bandwidth ** 2 * np.ones((Phi.T @ Phi).shape)) @ Phi.T @ q_i.T
        w_list.append(w.reshape(-1))
    w_all = np.array(w_list)

    mean_w = np.mean(w_all, axis=0)
    cov_w = np.cov(w_all.T)
    mean_traj = Phi @ mean_w
    std_traj = np.empty(time.shape)
    for i in range(nSteps):
        phi_i = Phi[i, :].transpose().reshape(1, -1)
        variance = bandwidth ** 2 * np.eye((phi_i @ phi_i.T).shape[0]) + phi_i @ cov_w @ phi_i.T
        std_traj[i] = np.sqrt(variance)

    plt.figure()
    plt.fill_between(time, mean_traj - 2 * std_traj, mean_traj + 2 * std_traj, alpha=0.5, edgecolor='#1B2ACC',
                     facecolor='#089FFF', label="std of desired trajectory")
    plt.plot(time, mean_traj, linewidth=2, color='#1B2ACC', label="mean of desired trajectory")
    plt.plot(time, q.T, alpha=0.3)
    plt.title('ProMP with ' + str(nBasis) + ' basis functions')
    plt.legend()

    #Conditioning
    if condition:
        y_d = 3
        Sig_d = 0.0002
        t_point = int(np.round(2300 / 2))

        tmp = cov_w @ Phi[t_point, :] / (Sig_d + Phi[t_point, :].T @ cov_w @ Phi[t_point, :])
        tmp = tmp.reshape(-1, 1)

        cov_w_new = cov_w - tmp @ Phi[t_point, :].reshape(1, -1) @ cov_w
        mean_w_new = mean_w + tmp @ (y_d - Phi[t_point, :].reshape(1, -1) @ mean_w)
        mean_traj_new = Phi @ mean_w_new
        std_traj_new = np.empty(time.shape)
        for i in range(nSteps):
            phi_i = Phi[i, :].transpose().reshape(1, -1)
            variance = bandwidth ** 2 * np.eye((phi_i @ phi_i.T).shape[0]) + phi_i @ cov_w_new @ phi_i.T
            std_traj_new[i] = np.sqrt(variance)

        plt.figure()
        plt.fill_between(time, mean_traj - 2 * std_traj, mean_traj + 2 * std_traj, alpha=0.5, edgecolor='#1B2ACC',
                         facecolor='#089FFF')
        plt.plot(time, mean_traj, color='#1B2ACC')
        plt.fill_between(time, mean_traj_new - 2 * std_traj_new, mean_traj_new + 2 * std_traj_new, alpha=0.5,
                         edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.plot(time, mean_traj_new, color='#CC4F1B')

        sample_traj = np.dot(Phi, np.random.multivariate_normal(mean_w_new, cov_w_new, 10).T)
        plt.plot(time, sample_traj, alpha=0.35)
        plt.title('ProMP after conditioning with new sampled trajectories')

    plt.draw_all()
    plt.pause(0.001)


if __name__ == "__main__":
    dt = 0.002
    time = np.arange(dt, 3, dt)
    nSteps = len(time)
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    nBasis = 30
    bandwidth = 0.2
    Phi = getProMPBasis(dt, nSteps, nBasis, bandwidth)

    w_list = []
    # compute w for each trajectory
    for q_i in q:
        q_i = q_i.reshape(1, -1)
        w = np.linalg.pinv(Phi.T @ Phi + bandwidth ** 2 * np.ones((Phi.T @ Phi).shape)) @ Phi.T @ q_i.T
        w_list.append(w.reshape(-1))
    w_all = np.array(w_list)

    mean_w = np.mean(w_all, axis=0)
    cov_w = np.cov(w_all.T)
    mean_traj = Phi @ mean_w
    std_traj = np.empty(time.shape)
    for i in range(nSteps):
        phi_i = Phi[i, :].transpose().reshape(1, -1)
        variance = bandwidth ** 2 * np.eye((phi_i @ phi_i.T).shape[0]) + phi_i @ cov_w @ phi_i.T
        std_traj[i] = np.sqrt(variance)

    plt.figure()
    plt.fill_between(time, mean_traj - 2 * std_traj, mean_traj + 2 * std_traj, alpha=0.5, edgecolor='#1B2ACC',
                     facecolor='#089FFF', label="std of desired trajectory")
    plt.plot(time, mean_traj, linewidth=2, color='#1B2ACC', label="mean of desired trajectory")
    plt.plot(time, q.T, alpha=0.3)
    plt.title('ProMP with ' + str(nBasis) + ' basis functions')
    plt.legend()

    y_d = 3
    Sig_d = 0.0002
    t_point = int(np.round(2300 / 2))

    tmp = cov_w @ Phi[t_point, :] / (Sig_d + Phi[t_point, :].T @ cov_w @ Phi[t_point, :])
    tmp = tmp.reshape(-1, 1)

    cov_w_new = cov_w - tmp @ Phi[t_point, :].reshape(1, -1) @ cov_w
    mean_w_new = mean_w + tmp @ (y_d - Phi[t_point, :].reshape(1, -1) @ mean_w)
    mean_traj_new = Phi @ mean_w_new
    std_traj_new = np.empty(time.shape)
    for i in range(nSteps):
        phi_i = Phi[i, :].transpose().reshape(1, -1)
        variance = bandwidth ** 2 * np.eye((phi_i @ phi_i.T).shape[0]) + phi_i @ cov_w_new @ phi_i.T
        std_traj_new[i] = np.sqrt(variance)

    plt.figure()
    plt.fill_between(time, mean_traj - 2 * std_traj, mean_traj + 2 * std_traj, alpha=0.5, edgecolor='#1B2ACC',
                     facecolor='#089FFF')
    plt.plot(time, mean_traj, color='#1B2ACC')
    plt.fill_between(time, mean_traj_new - 2 * std_traj_new, mean_traj_new + 2 * std_traj_new, alpha=0.5,
                     edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.plot(time, mean_traj_new, color='#CC4F1B')

    sample_traj = np.dot(Phi, np.random.multivariate_normal(mean_w_new, cov_w_new, 10).T)
    plt.plot(time, sample_traj, alpha=0.35)
    plt.title('ProMP after conditioning with new sampled trajectories')