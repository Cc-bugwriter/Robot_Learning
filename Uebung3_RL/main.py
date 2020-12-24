from src import load_file, linear_least_squares
import numpy as np
import math


def task_c(data_training):
    """
    sub task c in 3.td assignment
    :param data_training: array, [2 x n] training data set
    :return None:
    """
    input_data = data_training[0, :].reshape(1, -1)
    target_data = data_training[1, :].reshape(1, -1)

    pred_x = np.linspace(start=0.0, stop=6.0, num=int(6 / 0.01) + 1).reshape(1, -1)
    pred_y_lst = []

    feature_lst = [2, 3, 9]
    for feature in feature_lst:
        theta = linear_least_squares.fitting(input_data, target_data, feature)
        pred_y = linear_least_squares.prediction(pred_x, theta).reshape(1, -1)
        pred_y_lst.append(pred_y)

    linear_least_squares.visualisation(pred_x, pred_y_lst, feature_lst)


def task_d(data_training):
    """
    sub task d in 3.td assignment
    :param data_training: array, [2 x n] training data set
    :return None:
    """
    input_data = data_training[0, :].reshape(1, -1)
    target_data = data_training[1, :].reshape(1, -1)

    feature_lst = np.linspace(start=1, stop=9, num=9)
    rmse_lst = []
    for feature in feature_lst:
        theta = linear_least_squares.fitting(input_data, target_data, int(feature))
        pred_y = linear_least_squares.prediction(input_data, theta).reshape(1, -1)
        rmse = linear_least_squares.rmse(target_data, pred_y)
        rmse_lst.append(rmse)

    linear_least_squares.error_visualisation(rmse_lst, feature_lst)


def task_e(data_training, data_validation):
    """
    sub task e in 3.td assignment
    :param data_training: array, [2 x n] training data set
    :param data_validation: array, [2 x n] validation data set
    :return:
    """
    feature_lst = np.linspace(start=1, stop=9, num=9)

    input_data_train = data_training[0, :].reshape(1, -1)
    target_data_train = data_training[1, :].reshape(1, -1)

    input_data_val = data_validation[0, :].reshape(1, -1)
    target_data_val = data_validation[1, :].reshape(1, -1)

    rmse_lst_train = []
    rmse_lst_val = []
    for feature in feature_lst:
        theta = linear_least_squares.fitting(input_data_train, target_data_train, int(feature))
        pred_y_train = linear_least_squares.prediction(input_data_train, theta).reshape(1, -1)
        rmse_train = linear_least_squares.rmse(target_data_train, pred_y_train)
        rmse_lst_train.append(rmse_train)

        pred_y_val = linear_least_squares.prediction(input_data_val, theta).reshape(1, -1)
        rmse_val = linear_least_squares.rmse(target_data_val, pred_y_val)
        rmse_lst_val.append(rmse_val)

    linear_least_squares.model_selection(rmse_lst_train, rmse_lst_val, feature_lst)


if __name__ == "__main__":
    data_training = load_file.load(file="training_data.txt")
    data_validation = load_file.load(file="validation_data.txt")

    # task_c(data_training)
    #
    # task_d(data_training)

    task_e(data_training, data_validation)
