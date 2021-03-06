#!/usr/bin/env python
"""
File for all functions which read or generate data

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import keras
import numpy as np


def mnist(one_hot=False):
    """
    Load dataset from keras
    :param one_hot: convert to one hot
    """
    mnist = keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = np.where(x_train / 255 > 0.5, 0,1), np.where(x_test / 255 > 0.5, 0, 1)
    x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))

    if one_hot:
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test
