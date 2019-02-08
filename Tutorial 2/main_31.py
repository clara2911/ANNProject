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

    sin, square = generate_data.sin_square(verbose=verbose)

    data = sin  # use which dataset to train and test
    filter_output = False

    errors = []
    # number of RBF nodes in the network
    network_size_list =  [2, 5, 10, 20, 30, 40, 50, 63]  # NOTE: smaller than sample size
    for network_size in network_size_list:

        rbf_net = RBF_Net(network_size, data.train_X)

        y_train_pred, train_error = rbf_net.train(data.train_X, data.train_Y, method)

        y_pred, test_error = rbf_net.test(data.test_X, data.test_Y)

        # if you want to get perfect results for square, filter the output in the same manner as the data
        if (data == square and filter_output):
            y_pred = np.where(y_pred >= 0, 1, -1)
            test_error = rbf_net.calc_abs_res_error(data.test_Y, y_pred)

        print('# Nodes: ', network_size)
        print('Train error: ', train_error)
        print('Test error: ', test_error)
        errors.append(test_error)

    plotter.plot_errors(network_size_list, errors, title='Test error')

    plotter.plot_2d_function(data.train_X, data.train_Y, y_pred=y_train_pred, title='Train')
    plotter.plot_2d_function(data.test_X, data.test_Y, y_pred=y_pred, title='Test')


if __name__ == "__main__":
    part_3_1()
