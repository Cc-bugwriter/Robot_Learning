import numpy as np
from src import load_file, linear_least_squares, kernel_regression


def task_c(input_train, target_train, prep_input):
    """
    sub task c in 3.td assignment
    :param input_train:     array, [1, n] input vector of training set
    :param target_train:    array, [1, n] target vector of training set
    :param prep_input:      array, [1, m] input vector of prediction set
    :return None:
    """
    pred_y_lst = []

    feature_lst = [2, 3, 9]
    for feature in feature_lst:
        theta = linear_least_squares.fitting(input_train, target_train, feature)
        pred_y = linear_least_squares.prediction(pred_x, theta).reshape(1, -1)
        pred_y_lst.append(pred_y)

    linear_least_squares.visualisation(pred_x, pred_y_lst, feature_lst)


def task_d(input_train, target_train, feature_arr):
    """
    sub task d in 3.td assignment
    :param input_train:     array, [1, n] input vector of training set
    :param target_train:    array, [1, n] target vector of training set
    :param feature_arr:     array,  [k x int] list of different feature sizes
    :return None:
    """
    rmse_lst = []

    for feature in feature_arr:
        theta = linear_least_squares.fitting(input_train, target_train, int(feature))
        pred_y = linear_least_squares.prediction(input_train, theta).reshape(1, -1)
        rmse = linear_least_squares.rmse(target_train, pred_y)
        rmse_lst.append(rmse)

    linear_least_squares.error_visualisation(rmse_lst, feature_arr)


def task_e(input_train, target_train, input_val, target_val, feature_arr):
    """
    sub task e in 3.td assignment
    :param input_train:     array, [1, n] input vector of training set
    :param target_train:    array, [1, n] target vector of training set
    :param input_val:       array, [1, n] input vector of validation set
    :param target_val:      array, [1, n] target vector of validation set
    :param feature_arr:     array,  [k x int] list of different feature sizes
    :return:
    """
    rmse_lst_train = []
    rmse_lst_val = []
    for feature in feature_arr:
        theta = linear_least_squares.fitting(input_train, target_train, int(feature))
        pred_y_train = linear_least_squares.prediction(input_train, theta).reshape(1, -1)
        rmse_train = linear_least_squares.rmse(target_train, pred_y_train)
        rmse_lst_train.append(rmse_train)

        pred_y_val = linear_least_squares.prediction(input_val, theta).reshape(1, -1)
        rmse_val = linear_least_squares.rmse(target_val, pred_y_val)
        rmse_lst_val.append(rmse_val)

    linear_least_squares.model_selection(rmse_lst_train, rmse_lst_val, feature_arr)


def task_f(input_train, target_train, feature_arr):
    """
    sub task f in 3.td assignment
    :param input_train:    array, [1, n] input vector of training set
    :param target_train:   array, [1, n] target vector of training set
    :param feature_arr:    array,  [k x int] list of different feature sizes
    :return:
    """
    # compute mean and variance based on LOO
    rmse_mean, rmse_var = linear_least_squares.cross_validation(input_train, target_train, feature_arr)

    linear_least_squares.cv_visualization(rmse_mean, rmse_var, feature_arr)


def task_h(input_train, target_train, input_val, target_val, prep_x_kernel):
    """
    sub task c in 3.td assignment
    :param input_train:     array, [1, n] input vector of training set
    :param target_train:    array, [1, n] target vector of training set
    :param input_val:       array, [1, n] input vector of validation set
    :param target_val:      array, [1, n] target vector of validation set
    :param prep_x_kernel:   array, [1, m] input vector of prediction set
    :return None:
    """
    # plot Kernel Regression
    pred_y_kernel = kernel_regression.standard_kernel_regression(input_train, target_train, prep_x_kernel)
    kernel_regression.visualisation(prep_x_kernel, pred_y_kernel)

    # compare RMSE with best LLS model
    pred_y_kernel_val = kernel_regression.standard_kernel_regression(input_train, target_train, input_val)
    rmse_kernel = kernel_regression.rmse(target_val, pred_y_kernel_val)
    print("RMSE of Kernel Regression Model: {}".format(rmse_kernel))

    theta_lls = linear_least_squares.fitting(input_train, target_train, feature=3)
    pred_y_lls_val = linear_least_squares.prediction(input_val ,theta_lls)
    rmse_lls = linear_least_squares.rmse(target_val, pred_y_lls_val)
    print("RMSE of LSS Model: {}".format(rmse_lls))


if __name__ == "__main__":
    # data preparation
    data_training = load_file.load(file="training_data.txt")
    data_validation = load_file.load(file="validation_data.txt")

    input_data_train = data_training[0, :].reshape(1, -1)
    target_data_train = data_training[1, :].reshape(1, -1)

    input_data_val = data_validation[0, :].reshape(1, -1)
    target_data_val = data_validation[1, :].reshape(1, -1)

    features = np.linspace(start=1, stop=9, num=9)

    pred_x = np.linspace(start=0.0, stop=6.0, num=int(6 / 0.01) + 1).reshape(1, -1)

    # assignment subtask
    task_c(input_data_train, target_data_train, pred_x)

    task_d(input_data_train, target_data_train, features)

    task_e(input_data_train, target_data_train, input_data_val, target_data_val, features)

    task_f(input_data_train, target_data_train, features)

    task_h(input_data_train, target_data_train, input_data_val, target_data_val, pred_x)


