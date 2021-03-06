import numpy as np
import matplotlib.pyplot as plt


def load_file(file="spinbotdata.txt"):
    """
    load the local txt file as numpy operable data
    and process the data split
    :param file: str, name of local txt file
    :return:
        X: numpy array, [9, 100], input dataset
        Y: numpy array, [3, 100], target dataset
    """
    data = np.loadtxt(file)

    # split input data and target data
    X = data[:9, :]
    Y = data[9:, :]

    return X, Y


def regression(X, Y):
    """
    use least squares method to find the optimal parameter vectors
    :param X: [9, 100], input dataset
        where:
            q_1     : X[0, :]
            q_2     : X[1, :]
            q_3     : X[2, :]
            dq_1    : X[3, :]
            dq_2    : X[4, :]
            dq_3    : X[5, :]
            ddq_1   : X[6, :]
            ddq_2   : X[7, :]
            ddq_3   : X[8, :]
    :param Y: [3, 100], target dataset
    :return:
        theta[3, 1], parameter vectors
    """
    # build the designed feature matrix
    sample_times = X.shape[1]
    phi_design = np.zeros((3*sample_times, 3))
    for i in range(sample_times):
        phi = feature_matrix(X[:, i])
        phi_design[3*i: 3*(i+1), :] = phi

    # solve the optimization problem
    Y = Y.T.flatten().reshape(-1, 1)    # -> (3, 100) to (300, 1)

    theta = np.linalg.inv(phi_design.T @ phi_design) @ phi_design.T @ Y

    print("size of designed phi: {}".format(phi_design.shape))
    print("size of designed Y: {}".format(Y.shape))
    print("theta: {}".format(theta))

    return phi_design, theta


def feature_matrix(X_i):
    """
    build the feature matrix for each perception
    :param X_i: [9, ], input dataset of each detection
    :return:
            phi [3, 3], feature matrix as assignment 5.1 c
    """

    phi = np.array([
        [X_i[6], 0, 0],
        [X_i[6], 2.0 * X_i[5] * X_i[4] * X_i[2] + (X_i[2]**2) * X_i[7], X_i[8] - X_i[2] * (X_i[4]**2)],
        [1, 0, 0]
    ])
    # alternative contribution of feature matrix
    # phi = np.array([
    #     [X_i[6], 0, 0],
    #     [1.0, 0, 0],
    #     [0, 2.0 * X_i[4] * X_i[5] * X_i[2] + X_i[2]**2 * X_i[7], X_i[8] - X_i[2] * X_i[4]**2]
    # ])
    return phi.transpose()


def get_gravity(theta):
    """
    derive the unknown gravity from regression result
    :param theta:   [3, 1], optimal parameter vector from regression process
    :return:
        g: float, gravity
    """
    return theta[-1]/(theta[0] + theta[1])


def model_evaluation(theta, X, Y):
    """
    evaluate the regression result
    compare the prediction kraft/moment with the ground truth
    :param theta:   [3, 1], optimal parameter vector from regression process
    :param X: [9, 100], input dataset
    :param Y: [3, 100], target dataset
    :return:
    """
    X_transpose = X.T
    Y_pred = np.empty(Y.shape)
    for i, X_i in enumerate(X_transpose):
        phi_i = feature_matrix(X_i)
        y_pred_i = phi_i @ theta
        Y_pred[:, i] = y_pred_i.reshape(-1)

    time = np.linspace(0, Y.shape[1]-1, num=Y.shape[1]) / 500.0 * 1000.0
    for i in range(Y.shape[0]):
        plt.figure()
        plt.plot(time, Y[i, :], label="ground truth")
        plt.plot(time, Y_pred[i, :], label="prediction")
        plt.legend()
        fig_title = "joint {}".format(i + 1)
        plt.title(fig_title)
        plt.xlabel("time in ms")
        if i == 1:
            plt.ylabel("force in N")
        else:
            plt.ylabel("torque in Nm")
    plt.show()


if __name__ == "__main__":
    X_file, Y_file = load_file()
    phi_i = feature_matrix(X_file[:, 0])
    phi, theta = regression(X_file, Y_file)
    gravity = get_gravity(theta)
    print("the predicted gravity from dataset: {}".format(gravity))
    model_evaluation(theta, X_file, Y_file)


