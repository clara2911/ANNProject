#!/usr/bin/env python
"""
Main file for assignment 3.1
RBF Network on a sin and square function

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""
import numpy as np
import generate_data
import plotter
from rbf_NN import RBF_Net

verbose = True


def part_3_1():
    """
    Regression problem on two functions (sin, square) using
    batch learning with least squares
    """
    method = 'least_squares'
    network_size = 63 # number of RBF nodes in the network # NOTE: larger than sample size

    sin, square = generate_data.sin_square(verbose=verbose)

    data = sin  # use which dataset to train and test

    rbf_net = RBF_Net(network_size, data.train_X)

    y_train_pred, train_error = rbf_net.train(data.train_X, data.train_Y, method)

    y_pred, test_error = rbf_net.test(data.test_X, data.test_Y)

    print('Train error: ', train_error)
    print('Test error: ', test_error)

    plotter.plot_2d_function(data.train_X, data.train_Y, y_pred=y_train_pred, title='Train')
    plotter.plot_2d_function(data.test_X, data.test_Y, y_pred=y_pred, title='Test')


if __name__ == "__main__":
    part_3_1()
