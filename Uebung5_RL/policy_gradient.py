from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import time

# %matplotlib inline
np.set_printoptions(precision=3, linewidth=100000)

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10


# YOUR CODE HERE
def policy_gradient(mean=0.0, std=10.0, alfa=0.8, episodes=25, maxIter=100, numDim=10, env=Pend2dBallThrowDMP(),
                    trick=False, variance_var=False):
    """
    Optimize the upper-level policy parameters using the policy gradient.
    :param mean:        float, initial mean of policy parameters            default: 0.0
    :param std:         float, fixed std of policy parameters               default: 10.0
    :param alfa:        float, learning rate                                default: 0.8
    :param episodes:    int, size of sample                                 default: 25
    :param maxIter:     int, max iterations of policy updates               default: 100
    :param numDim:      int, dimension of weights vector in DMP model       default: 10
    :param env:         Pend2dBallThrowDMP, DMP model                       default: Pend2dBallThrowDMP()
    :param trick:       boolean, subtracting a baseline from the gradient   default: False
    :param variance_var:    boolean, update the variance,                   default: False
    :return mue:        array [10, _], learned mean of weights
    """
    mue = mean * np.ones(numDim)
    std = std * np.ones(numDim)

    theta = np.empty((episodes, numDim))
    R = np.empty((maxIter, episodes))
    R_mean = np.empty(maxIter)

    for t in range(maxIter):
        # sample
        for i in range(episodes):
            # obtain weights
            for j, mue_i in enumerate(mue):
                theta[i, j] = np.random.normal(mue_i, std[j])
            # obtain rewards
            R[t, i] = env.getReward(theta[i])

        # obtain mean of rewards in all episodes
        R_mean[t] = np.mean(R[t])                   # -> baseline in trick operation

        # logarithm derivation
        sig_inv = 1 / std * np.ones(numDim)         # -> diagonal element vector
        cov_inv = 1 / std ** 2 * np.ones(numDim)    # -> diagonal element vector
        tri_inv = 1 / std ** 3 * np.ones(numDim)    # -> diagonal element vector

        delta_log_mue = (theta - mue) @ np.diag(cov_inv)
        if not trick:
            delta_log_mue *= R[t].reshape(-1, 1)
            # for i in range(episodes):
            #     delta_log_mue[i] = np.diag(cov_inv) @ (theta[i] - mue.reshape(-1)) * R[t, i]
        else:
            delta_log_mue *= R[t].reshape(-1, 1) - R_mean[t]
            # for i in range(episodes):
            #     delta_log_mue[i] = cov_inv * (theta[i] - mue.reshape(-1)) * (R[t, i] - R_mean[t])

        w_J_mue = np.sum(delta_log_mue, axis=0) / float(episodes)
        # update mean
        mue += alfa * w_J_mue[:10]

        if variance_var:
            delta_log_std = tri_inv * (theta - mue) ** 2 - sig_inv
            delta_log_std *= R[t].reshape(-1, 1) - R_mean[t]

            w_J_std = np.sum(delta_log_std, axis=0) / float(episodes)
            # update std
            std += alfa * w_J_std[:10]
            std[std < 1.0] = 1.0    # -> minimal threshold for std update

    return mue, R_mean


def plot_average(mue_mean, mue_std, title, label=None, numDim=100):
    """
    plot the mean of the average return of all runs with 95% confidence
    :param mue_mean:    mean of the average
    :param mue_std:     std of the average
    :param title:       title of figure
    :param label:       str, label of input data                            default: None
    :param numDim:      int, dimension of weights vector in DMP model       default: 100
    :return:
    """
    x = np.arange(numDim)
    mue_mean = mue_mean.reshape((-1, numDim))
    mue_std = mue_std.reshape((-1, numDim))

    plt.figure()
    for i in range(mue_mean.shape[0]):
        plt.plot(x, mue_mean[i])
        if label is None:
            plt.fill_between(x, mue_mean[i] - 2 * mue_std[i], mue_mean[i] + 2 * mue_std[i], alpha=0.2)
        else:
            plt.fill_between(x, mue_mean[i] - 2 * mue_std[i], mue_mean[i] + 2 * mue_std[i], alpha=0.2, label=label[i])
    plt.title(title)
    plt.xlabel("iteration index")
    plt.ylabel("reward value in log scale")
    if label is not None:
        plt.legend()
    plt.show()


def plot_preprocess(rewards):
    """
    preprocess in scale transform (linear -> log)
    :param rewards:     [10, n], array, rewards in each iteration
    :return mean:       [n, _], array, mean of rewards in log scale
            std:        [n, _], array, std of rewards in log scale
    """
    # linear to log scale
    reward_sign = np.sign(rewards)
    rewards = reward_sign * np.log(np.abs(rewards))

    mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)

    # # linear to log scale
    # reward_sign = np.sign(mean)
    # mean = reward_sign * np.log(np.abs(mean))
    # std = np.log(std)

    return mean, std

# task 5.3 b
reward_lst = []
for i in range(numTrials):
    mue, reward = policy_gradient(mean=0.0, std=10.0, alfa=0.1, episodes=numSamples, maxIter=maxIter, numDim=numDim,
                                  env=env)
    reward_lst.append(reward)

reward_arr = np.array(reward_lst)
reward_mean, reward_std = plot_preprocess(reward_arr)
title = "fixed std without baseline subtracting"
label = ["alpha = 0.1"]
plot_average(reward_mean, reward_std, title, label)

# task 5.3 c
reward_lst = []
for i in range(numTrials):
    mue, reward = policy_gradient(mean=0.0, std=10.0, alfa=0.1, episodes=numSamples, maxIter=maxIter, numDim=numDim,
                                  env=env, trick=True)
    reward_lst.append(reward)

reward_arr = np.array(reward_lst)
reward_mean, reward_std = plot_preprocess(reward_arr)
title = "fixed std with baseline subtracting trick"
label = ["alpha = 0.1"]
plot_average(reward_mean, reward_std, title, label)

# task 5.3 d
reward_lst = []
for i in range(numTrials):
    mue, reward = policy_gradient(mean=0.0, std=10.0, alfa=0.4, episodes=numSamples, maxIter=maxIter, numDim=numDim,
                                  env=env, trick=True)
    reward_lst.append(reward)
for i in range(numTrials):
    mue, reward = policy_gradient(mean=0.0, std=10.0, alfa=0.2, episodes=numSamples, maxIter=maxIter, numDim=numDim,
                                  env=env, trick=True)
    reward_lst.append(reward)

reward_arr = np.array(reward_lst)
reward_mean_high, reward_std_high = plot_preprocess(reward_arr[:10, :])
reward_mean_low, reward_std_low = plot_preprocess(reward_arr[10:, :])
title = "Learning Rate contrast"
label = ["alpha = 0.4", "alpha = 0.2"]
reward_mean = np.vstack((reward_mean_high, reward_mean_low))
reward_std = np.vstack((reward_std_high, reward_std_low))
plot_average(reward_mean, reward_std, title, label)

# task 5.3 e  -> run with 5.3 d together
for i in range(numTrials):
    mue, reward = policy_gradient(mean=0.0, std=10.0, alfa=0.4, episodes=numSamples, maxIter=maxIter, numDim=numDim,
                                  env=env, trick=True, variance_var=True)
    reward_lst.append(reward)

reward_arr = np.array(reward_lst)
reward_mean_var, reward_std_var = plot_preprocess(reward_arr[20:, :])
reward_mean = np.vstack((reward_mean_high, reward_mean_var))
reward_std = np.vstack((reward_std_high, reward_std_var))
title = "Learning Rate contrast with alpha = 0.4"
label = ["fixed variance", "variable variance"]
plot_average(reward_mean, reward_std, title, label)