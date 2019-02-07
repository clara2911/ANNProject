#!/usr/bin/env python
"""
Main file for assignment 3.2
RBF Network on a sin and square function

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""
import numpy as np
import generate_data
import plotter
from ann import ANN
from rbf_NN import RBF_Net

verbose = True


def part_3_2():
    """
    Regression problem on two functions (sin, square) with added noise
    using online (sequential) learning with delta rule
    """
    method = 'delta_rule'  # delta_rule
    network_size = 63
    learning_rate = 0.01  # optimal value: 0.01 bigger overshoots, smaller underperforms

    sin, square = generate_data.sin_square(verbose=verbose)
    sin = add_noise_to_data(sin)
    square = add_noise_to_data(square)

    data = sin  # use which dataset to train and test

    rbf_net = RBF_Net(network_size, data.train_X)

    y_train_pred, train_error = rbf_net.train(data.train_X, data.train_Y, method, lr=learning_rate)

    y_pred, test_error = rbf_net.test(data.test_X, data.test_Y)

    print('Train error: ', train_error)
    print('Test error: ', test_error)

    plotter.plot_2d_function(data.train_X, data.train_Y, y_pred=y_train_pred, title='Train')
    plotter.plot_2d_function(data.test_X, data.test_Y, y_pred=y_pred, title='Test')


def compare_to_ann():
    """
    Regression problem on two functions (sin, square) with added noise
    using batch learning with a 2 layer neural network
    """

    epochs = 1000
    hidden_neurons = 63  # same number as RBF nodes
    output_neurons = 1

    sin, square = generate_data.sin_square(verbose=verbose)
    sin = add_noise_to_data(sin)
    square = add_noise_to_data(square)

    data = sin  # use which dataset to train and test

    batch_size = data.train_X.shape[0]  # set batch size equal to number of data points

    ann = ANN(epochs, batch_size, hidden_neurons, output_neurons)

    y_pred = ann.solve(sin.train_X, sin.train_Y, sin.test_X, sin.test_Y)

    plotter.plot_2d_function(sin.train_X, sin.train_Y, y_pred=y_pred)


def add_noise_to_data(data):
    """
    Add noise to all the train and test data samples
    """
    data.train_X = data.train_X + np.random.normal(0, 0.1, data.train_X.shape)  # zero mean, 0.1 std
    data.train_Y = data.train_Y + np.random.normal(0, 0.1, data.train_Y.shape)
    data.test_X = data.test_X + np.random.normal(0, 0.1, data.test_X.shape)
    data.test_Y = data.test_Y + np.random.normal(0, 0.1, data.test_Y.shape)
    return data


if __name__ == "__main__":
    part_3_2()
    # compare_to_ann()
