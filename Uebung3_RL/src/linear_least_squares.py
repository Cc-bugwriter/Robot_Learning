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
        plt.plot(pred_x.reshape(-1), pred_y.reshape(-1), label="features: {}".format(feature_lst[i]), linewidth=2)
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


def cross_validation(input_data, target_data, feature_arr):
    """
    implement leave-one-out cross validation
    :param input_data:      array [1 x n], vector x
    :param target_data:     array [1 x n], vector y
    :param feature_arr:     array [m, ], number of selective features
    :return rmse_mean:      array [m, ], mean of the RMSE for the learned model
            rmse_var:       array [m, ], variance of the RMSE for the learned model
    """
    # initial
    rmse_mean = np.empty(feature_arr.shape)
    rmse_var = np.empty(feature_arr.shape)

    for i, feature in enumerate(feature_arr):
        # initial intermediate variables
        rmse_lst = []

        for index_out in range(input_data.shape[1]):
            # split data
            input_data_rest = np.delete(input_data, (0, index_out)).reshape(1, -1)
            target_data_rest = np.delete(target_data, (0, index_out)).reshape(1, -1)
            input_data_out = np.array([input_data[0, index_out]]).reshape(1, -1)
            target_data_out = np.array([target_data[0, index_out]]).reshape(1, -1)

            # predict and compute rmse based on out data
            theta_rest = fitting(input_data_rest, target_data_rest, int(feature))
            pred_out = prediction(input_data_out, theta_rest)
            rmse_out = rmse(target_data_out, pred_out)

            rmse_lst.append(rmse_out)

        rmse_arr = np.array(rmse_lst)

        rmse_mean[i] = rmse_arr.mean()
        rmse_var[i] = rmse_arr.var()

    return rmse_mean, rmse_var


def cv_visualization(rmse_mean, rmse_var, feature_arr):
    """
    visualise the mean and variance of each model error(RMSE) in single figure
    :param rmse_mean:   array [m, ], mean of the RMSE for the learned model
    :param rmse_var:    array [m, ], variance of the RMSE for the learned model
    :param feature_arr: array [m, ], number of selective features
    :return:
    """
    plt.figure()
    line, = plt.plot(feature_arr, rmse_mean, linewidth=2, color='red')
    area = plt.fill_between(feature_arr, rmse_mean + rmse_var, rmse_mean - rmse_var,
                            color='green', alpha=0.2)
    plt.title("mean/ variance of RMSE (LOO) vs. features")
    plt.xlabel("number of features")
    plt.ylabel("RMSE of LOO")
    plt.legend([line, area], ["mean", "variance"])
    plt.grid(True)
    plt.show()

