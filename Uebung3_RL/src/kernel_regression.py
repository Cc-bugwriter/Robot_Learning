import numpy as np
import matplotlib.pyplot as plt


def squared_kernel(x_i, x_j, sigma=0.15):
    """
    generate an exponential squared kernel with input vectors
    :param x_i:     array,  [1, n] first input vector
                or  float,  first input value
    :param x_j:     array,  [1, n] second input vector
                or  float,  second input value
    :param sigma:   float, kernel coefficient (default value: 0.15)
    :return:
    """
    # preprocess
    if x_i.shape is not None:
        x_i = x_i.reshape(1, -1)
    if x_j.shape is not None:
        x_j = x_j.reshape(1, -1)

    # compute a kernel
    if x_i.shape is None and x_j.shape is None:
        kernel = np.exp(-(1 / sigma ** 2) * (np.abs(x_i - x_j) ** 2))
    else:
        kernel = np.exp(-(1 / sigma ** 2) * np.square(np.linalg.norm(x_i - x_j, axis=0, ord=2))).reshape(x_i.shape)

    return kernel


def standard_kernel_regression(input_data, target_data, pred_data):
    """
    predict with kernel, which base on training dataset
    :param input_data:      array [1 x n], training vector x
    :param target_data:     array [1 x n], training target y
    :param pred_data:       array [1 x m], prediction input vector
    :return pred_result:    array [1 x m], prediction result vector
    """
    # initial
    K = np.empty((input_data.shape[1], input_data.shape[1]))
    k = np.empty((input_data.shape[1], pred_data.shape[1]))

    # compute Kernel Coefficient
    for i in range(input_data.shape[1]):
        for j in range(input_data.shape[1]):
            K[i, j] = squared_kernel(input_data[0, i], input_data[0, j])
        k[i, :] = squared_kernel(pred_data, input_data[0, i])

    # predict
    pred_result = k.T @ np.linalg.inv(K) @ target_data.T

    return pred_result.reshape(1, -1)


def visualisation(pred_x, pred_y):
    """
    plot the resulting prediction in single figure
    :param pred_x:          array, [1, n] input vector to prediction
    :param pred_y:          array, [1, n] prediction vectors
    :return:
    """
    plt.figure()
    plt.plot(pred_x.reshape(-1), pred_y.reshape(-1), linewidth=2)
    plt.title("resulting predictions with standard kernel regression")
    plt.xlabel("input value")
    plt.ylabel("prediction value")
    plt.grid(True)
    plt.show()


def rmse(target_y, pred_y):
    """
    compute the root mean square error between prediction value and ground truth value (copy from LLS)
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
