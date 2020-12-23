import numpy as np
import os


def load(path='data_ml', file='training_data.txt'):
    """
    load local txt file, convert them in array as dataset
    :param path:    str, path of locally saved data
    :param file:    str, file name of txt
    :return data:   array [2 x n], array of dataset
        first row:  vector x
        second row: vector y
    """
    local_path = os.getcwd()
    file_path = os.path.join(local_path, path, file)
    return np.loadtxt(file_path, dtype=str)