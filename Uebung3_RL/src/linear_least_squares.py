import numpy as np
import math
import matplotlib.pyplot as plt


def fitting(input_data, target_data, feature=2):
    """
    implement fitting process with linear least squares methode
    :param input_data:      array [1 x n], vector x
    :param target_data:     array [1 x n], vector y
    :param feature:         int m, number of selective features (default value: 2)
    :return theta:          array, [m x 1], weighting coefficient array of features elements
    """
    # initial
    phi = np.empty((feature, input_data.shape[-1]))

    # assign feature agency
    for index, row in enumerate(phi):
        for i, x in enumerate(input_data[0]):
            row[i] = math.sin(2 ** index * x)
        phi[index, :] = row.reshape(1, -1)

    # linear least squares methode
    theta_T = target_data @ np.linalg.pinv(phi.T @ phi) @ phi.T
    theta = theta_T.T
    return theta


def prediction(pred_x, theta):
    """
    prediction input with LLS coefficient
    :param pred_x:  array, [1 x n]  input values to prediction
    :param theta:   array, [m x 1]  weight coefficient of feature agency
    :return pred_y: array, [1 x n]  prediction result
    """
    # initial
    phi = np.empty((theta.shape[0], pred_x.shape[-1]))

    # assign feature agency
    for index, row in enumerate(phi):
        for i, x in enumerate(pred_x[0]):
            row[i] = row[i] = math.sin(2 ** index * x)
        phi[index, :] = row.reshape(1, -1)

    # predict
    pred_y = theta.T @ phi

    return pred_y


def visualisation(pred_x, pred_y_lst, feature_lst):
    """
    plot the resulting prediction in single figure
    :param pred_x:          array, [1, n] input vector to prediction
    :param pred_y_lst:      list,  [k x array] prediction vectors with different feature sizes
    :param feature_lst:     list,  [k x int] list of different feature sizes
    :return:
    """
    plt.figure()
    for i, pred_y in enumerate(pred_y_lst):
        plt.plot(pred_x.reshape(-1), pred_y.reshape(-1), label=feature_lst[i], linewidth=2)
    plt.title("resulting predictions with different features")
    plt.xlabel("input value")
    plt.ylabel("prediction value")
    plt.legend()
    plt.grid(True)
    plt.show()


def rmse(target_y, pred_y):
    """
    compute the root mean square error between prediction value and ground truth value
    :param target_y:    array, [1, n] ground truth of labeled dataset
    :param pred_y:      array, [1, n] predicted value
    :return error:      float, root mean square error
    """
    # initial
    error = 0

    # compute actual root mean square error
    diff = target_y - pred_y
    error = np.sqrt(1/pred_y.shape[1] * np.sum(diff * diff))

    return error


def error_visualisation(error_lst, feature_lst, label=None):
    """
    plot the error under defined feature size in single figure
    :param error_lst:   list,  [k x float] list of different root mean square errors
    :param feature_lst: list,  [k x int] list of different feature sizes
    :param label: str,  the label of prediction result (default value: None)
    :return:
    """
    if label is None:
        plt.figure()
        plt.plot(feature_lst, error_lst, linewidth=2)
    else:
        plt.plot(feature_lst, error_lst, linewidth=2, label=label)
    plt.title("RMSE vs. number of features")
    plt.xlabel("number of features")
    plt.ylabel("RMSE")
    plt.grid(True)

    if label is None:
        plt.show()


def model_selection(error_lst_train, error_lst_val, feature_lst):
    """
    plot RMSE of training set and validation set in single figure
    :param error_lst_train: list,  [k x float] list of different root mean square errors in tr
    :param error_lst_val:   list,  [k x float] list of different root mean square errors
    :param feature_lst:     list,  [k x int] list of different feature sizes
    :return:
    """
    plt.figure()
    error_visualisation(error_lst_train, feature_lst, label="training error")
    error_visualisation(error_lst_val, feature_lst, label="validation error")
    plt.legend()
    plt.show()
