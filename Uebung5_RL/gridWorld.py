import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


##
def genGridWorld():
    O = -1e5  # Dangerous places to avoid
    D = 35  # Dirt
    W = -100  # Water
    C = -3000  # Cat
    T = 1000  # Toy
    grid_list = {0: '', O: 'O', D: 'D', W: 'W', C: 'C', T: 'T'}
    grid_world = np.array([
        [0, O, O, 0, 0, O, O, 0, 0, 0],
        [0, 0, 0, 0, D, O, 0, 0, D, 0],
        [0, D, 0, 0, 0, O, 0, 0, O, 0],
        [O, O, O, O, 0, O, 0, O, O, O],
        [D, 0, 0, D, 0, O, T, D, 0, 0],
        [0, O, D, D, 0, O, W, 0, 0, 0],
        [W, O, 0, O, 0, O, D, O, O, 0],
        [W, 0, 0, O, D, 0, 0, O, D, 0],
        [0, 0, 0, D, C, O, 0, 0, D, 0]])
    return grid_world, grid_list


##
def showWorld(grid_world, tlt):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(tlt)
    ax.set_xticks(np.arange(0.5, 10.5, 1))
    ax.set_yticks(np.arange(0.5, 9.5, 1))
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.imshow(grid_world, interpolation='nearest', cmap='copper')
    return ax


##
def showTextState(grid_world, grid_list, ax):
    for x in range(grid_world.shape[0]):
        for y in range(grid_world.shape[1]):
            if grid_world[x, y] >= -3000:
                ax.annotate(grid_list.get(grid_world[x, y]), xy=(y, x), horizontalalignment='center')


##
def showPolicy(policy, ax):
    for x in range(policy.shape[0]):
        for y in range(policy.shape[1]):
            if policy[x, y] == 0:
                ax.annotate('$\downarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 1:
                ax.annotate(r'$\rightarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 2:
                ax.annotate(r'$\uparrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 3:
                ax.annotate('$\leftarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 4:
                ax.annotate('$\perp$', xy=(y, x), horizontalalignment='center')


##
def ValIter(R, discount=0.8, maxSteps=15, infHor=False, probModel=None):
    """
    Value Iteration algorithm
    :param R:           [9, 10] array, grid_world
    :param discount:    float,  discount factor in infinite horizon model   default: 0.8
    :param maxSteps:    int,    maximal step in finite horizon model        default: 15
    :param infHor:      bool,   model switch:                               default: False
                                ture -> infinite horizon model
                                false ->  finite horizon model
    :param probModel:   tulpe,  probabilistic transition,                   default: None
    :return:
    """
    # YOUR CODE HERE
    h, w = R.shape

    if not infHor:  # -> finite horizon model, task 5.2 a
        V = np.zeros((h, w, maxSteps))
        A = 4 * np.ones((h, w, maxSteps))
        if probModel is None:
            # task 5.2 a
            for T in reversed(range(maxSteps)):
                if T == maxSteps - 1:
                    # last layer
                    V[:, :, T] = grid_world
                else:
                    V[:, :, T], A[:, :, T] = maxAction(V[:, :, T + 1], R, 1.0)
        else:
            # task 5.2 d
            for T in reversed(range(maxSteps)):
                if T == maxSteps - 1:
                    # last layer
                    V[:, :, T] = grid_world
                else:
                    V[:, :, T], A[:, :, T] = maxAction(V[:, :, T + 1], R, 1.0, probModel)

    if infHor:  # -> infinite horizon model, task 5.2 c
        V = grid_world.reshape((h, w, 1))
        A = 4 * np.ones((h, w, 1))
        while infHor:
            V_back, A_back = maxAction(V[:, :, 0], R, discount)

            V = np.concatenate((V_back.reshape((h, w, 1)), V), axis=2)
            A = np.concatenate((A_back.reshape((h, w, 1)), A), axis=2)

            # break loop if convergence
            delta = np.sum(abs(V_back - V[:, :, 1]))
            if delta < 1e-5:
                break

    return V, A


##
def maxAction(V, R, discount, probModel=None):
    """
    V-Function for time step t:
                                argmax -> action
                                max -> value
    :param V:   [9, 10] array,  V-Function for last time step
    :param R:   [9, 10] array,  grid_world
    :param discount:    float,  discount factor in infinite horizon model
    :param probModel:   tulpe,  probabilistic transition, default: None
    :return:
    """
    # YOUR CODE HERE
    h, w = R.shape
    V_tmp = V.copy()
    A_tmp = 4 * np.ones(V_tmp.shape)

    if probModel is None:
        # grid search of s
        for i in range(h):
            for j in range(w):
                # candidates dict
                a_candidates = {0: (i + 1, j), 1: (i, j + 1), 2: (i - 1, j), 3: (i, j - 1), 4: (i, j)}
                if i == 0:
                    del a_candidates[2]
                if i == h - 1:
                    del a_candidates[0]
                if j == 0:
                    del a_candidates[3]
                if j == w - 1:
                    del a_candidates[1]

                # default as stay
                Q_max = R[i, j]
                A = 4
                for a_i in a_candidates:
                    Q = discount * V[a_candidates[a_i]]
                    if Q > Q_max:
                        Q_max = Q
                        A = a_i
                    # Q_max = Q if Q > Q_max else Q_max
                V_tmp[i, j] = R[i, j] + Q_max
                A_tmp[i, j] = A
    else:
        for i in range(h):
            for j in range(w):
                # candidates dict
                a_candidates = {0: (i + 1, j), 1: (i, j + 1), 2: (i - 1, j), 3: (i, j - 1), 4: (i, j)}

                # default as stay
                Q_max = R[i, j]
                A = 4
                for a_i in a_candidates:
                    # here enumerate all permutation of action pairs
                    if a_i == 0:
                        # point
                        if i == 0 and j == 0:
                            Q = probModel[0] * V[a_candidates[0]] + probModel[1] * V[a_candidates[1]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == 0 and j == w - 1:
                            Q = probModel[0] * V[a_candidates[0]] + probModel[1] * V[a_candidates[3]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1 and j == 0:
                            Q = probModel[1] * V[a_candidates[1]] + \
                                (probModel[0] + probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1 and j == w - 1:
                            Q = probModel[1] * V[a_candidates[3]] + \
                                (probModel[0] + probModel[1] + probModel[2]) * V[a_candidates[4]]
                        # edge
                        elif i == h - 1:
                            Q = probModel[1] * V[a_candidates[1]] + probModel[1] * V[a_candidates[3]] + \
                                (probModel[0] + probModel[2]) * V[a_candidates[4]]
                        elif j == 0:
                            Q = probModel[0] * V[a_candidates[0]] + probModel[1] * V[a_candidates[1]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif j == w - 1:
                            Q = probModel[0] * V[a_candidates[0]] + probModel[1] * V[a_candidates[3]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        # general
                        else:
                            Q = probModel[0] * V[a_candidates[0]] + probModel[1] * V[a_candidates[1]] + \
                                probModel[1] * V[a_candidates[3]] + probModel[2] * V[a_candidates[4]]
                    if a_i == 1:
                        # point
                        if i == 0 and j == 0:
                            Q = probModel[0] * V[a_candidates[1]] + probModel[1] * V[a_candidates[0]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == 0 and j == w - 1:
                            Q = probModel[1] * V[a_candidates[0]] + \
                                (probModel[0] + probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1 and j == 0:
                            Q = probModel[0] * V[a_candidates[1]] + probModel[1] * V[a_candidates[2]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1 and j == w - 1:
                            Q = probModel[1] * V[a_candidates[2]] + \
                                (probModel[0] + probModel[1] + probModel[2]) * V[a_candidates[4]]
                        # edge
                        elif j == w - 1:
                            Q = probModel[1] * V[a_candidates[0]] + probModel[1] * V[a_candidates[2]] + \
                                (probModel[0] + probModel[2]) * V[a_candidates[4]]
                        elif i == 0:
                            Q = probModel[0] * V[a_candidates[1]] + probModel[1] * V[a_candidates[0]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1:
                            Q = probModel[0] * V[a_candidates[1]] + probModel[1] * V[a_candidates[2]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        # general
                        else:
                            Q = probModel[0] * V[a_candidates[1]] + probModel[1] * V[a_candidates[0]] + \
                                probModel[1] * V[a_candidates[2]] + probModel[2] * V[a_candidates[4]]
                    if a_i == 2:
                        # point
                        if i == 0 and j == 0:
                            Q = probModel[1] * V[a_candidates[1]] + \
                                (probModel[0] + probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == 0 and j == w - 1:
                            Q = probModel[1] * V[a_candidates[3]] + \
                                (probModel[0] + probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1 and j == 0:
                            Q = probModel[0] * V[a_candidates[2]] + probModel[1] * V[a_candidates[1]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1 and j == w - 1:
                            Q = probModel[0] * V[a_candidates[2]] + probModel[1] * V[a_candidates[3]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        # edge
                        elif i == 0:
                            Q = probModel[1] * V[a_candidates[1]] + probModel[1] * V[a_candidates[3]] + \
                                (probModel[0] + probModel[2]) * V[a_candidates[4]]
                        elif j == 0:
                            Q = probModel[0] * V[a_candidates[2]] + probModel[1] * V[a_candidates[1]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif j == w - 1:
                            Q = probModel[0] * V[a_candidates[2]] + probModel[1] * V[a_candidates[3]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        # general
                        else:
                            Q = probModel[0] * V[a_candidates[2]] + probModel[1] * V[a_candidates[1]] + \
                                probModel[1] * V[a_candidates[3]] + probModel[2] * V[a_candidates[4]]
                    if a_i == 3:
                        # point
                        if i == 0 and j == 0:
                            Q = probModel[1] * V[a_candidates[0]] + \
                                (probModel[0] + probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == 0 and j == w - 1:
                            Q = probModel[0] * V[a_candidates[3]] + probModel[1] * V[a_candidates[0]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1 and j == 0:
                            Q = probModel[1] * V[a_candidates[2]] + \
                                (probModel[0] + probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1 and j == w - 1:
                            Q = probModel[0] * V[a_candidates[3]] + probModel[1] * V[a_candidates[2]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        # edge
                        elif j == 0:
                            Q = probModel[1] * V[a_candidates[0]] + probModel[1] * V[a_candidates[2]] + \
                                (probModel[0] + probModel[2]) * V[a_candidates[4]]
                        elif i == 0:
                            Q = probModel[0] * V[a_candidates[3]] + probModel[1] * V[a_candidates[0]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        elif i == h - 1:
                            Q = probModel[0] * V[a_candidates[3]] + probModel[1] * V[a_candidates[2]] + \
                                (probModel[1] + probModel[2]) * V[a_candidates[4]]
                        else:
                            Q = probModel[0] * V[a_candidates[3]] + probModel[1] * V[a_candidates[0]] + \
                                probModel[1] * V[a_candidates[2]] + probModel[2] * V[a_candidates[4]]
                    if a_i == 4:
                        Q = R[i, j]

                    if Q > Q_max:
                        Q_max = Q
                        A = a_i

                V_tmp[i, j] = R[i, j] + Q_max
                A_tmp[i, j] = A

    return V_tmp, A_tmp


##
# def findPolicy(V, R, discount, probModel=None):
#     """
#     V-Function for time step t:
#                                 argmax -> action
#                                 max -> value
#     :param V:   [9, 10] array,  V-Function for last time step
#     :param R:   [9, 10] array,  grid_world
#     :param discount:    float,  discount factor in infinite horizon model
#     :param probModel:   tulpe,  probabilistic transition, default: None
#     :return:
#     """
#     # YOUR CODE HERE
#     return 0
############################

saveFigures = False

data = genGridWorld()
grid_world = data[0]
grid_list = data[1]

# YOUR CODE HERE
probModel = (0.7, 0.1, 0.1)
#
ax = showWorld(grid_world, 'Environment')
showTextState(grid_world, grid_list, ax)
if saveFigures:
    plt.savefig('gridworld.pdf')

# Finite Horizon
V_all, A_all = ValIter(grid_world, maxSteps=15)
V = V_all[:, :, 0]
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon')
if saveFigures:
    plt.savefig('value_Fin_15.pdf')

# policy = findPolicy(...)
ax = showWorld(grid_world, 'Policy - Finite Horizon' + ' Step size: 10')
showPolicy(A_all[:, :, 0], ax)
if saveFigures:
    plt.savefig('policy_Fin_15.pdf')

# Infinite Horizon
V_all, A_all = ValIter(grid_world, discount=0.8, infHor=True)
V = V_all[:, :, 0]
showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
if saveFigures:
    plt.savefig('value_Inf_08.pdf')

# policy = findPolicy(...);
ax = showWorld(grid_world, 'Policy - Infinite Horizon')
showPolicy(A_all[:, :, 0], ax)
if saveFigures:
    plt.savefig('policy_Inf_08.pdf')

# Finite Horizon with Probabilistic Transition
V_all, A_all = ValIter(grid_world, maxSteps=15, probModel=probModel)
V = V_all[:, :, 0]
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
if saveFigures:
    plt.savefig('value_Fin_15_prob.pdf')

# policy = findPolicy(...)
ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
showPolicy(A_all[:, :, 0], ax)
if saveFigures:
    plt.savefig('policy_Fin_15_prob.pdf')
