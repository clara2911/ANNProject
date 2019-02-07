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
    method = 'delta_rule'
    network_size = 100

    sin, square = generate_data.sin_square(verbose=verbose)
    sin = add_noise_to_data(sin)
    square = add_noise_to_data(square)

    rbf_net = RBF_Net(network_size, sin.train_X)

    y_pred = rbf_net.train(sin.train_X, sin.train_Y, method)


def add_noise_to_data(data):
    """
    Add noise to all the train and test data samples
    """
    data.train_X = data.train_X + np.random.normal(0, 0.1, data.train_X.shape)  # zero mean, 0.1 std #generate_data.add_noise(data.train_X)
    data.train_Y = data.train_Y + np.random.normal(0, 0.1, data.train_Y.shape) #generate_data.add_noise(data.train_Y)
    data.test_X = data.test_X + np.random.normal(0, 0.1, data.test_X.shape) #generate_data.add_noise(data.test_X)
    data.test_Y = data.test_Y + np.random.normal(0, 0.1, data.test_Y.shape) # generate_data.add_noise(data.test_Y)
    return data


def compare_to_ann():
    """
    Regression problem on two functions (sin, square) with added noise
    using batch learning with a 2 layer neural network
    """

    epochs = 1000
    batch_size = 10
    hidden_neurons = 10
    output_neurons = 1

    sin, square = generate_data.sin_square(verbose=verbose)
    sin = add_noise_to_data(sin)
    square = add_noise_to_data(square)

    ann = ANN(epochs, batch_size, hidden_neurons, output_neurons)

    y_pred = ann.solve(sin.train_X, sin.train_Y, sin.test_X, sin.test_Y)

    plotter.plot_2d_function(sin.train_X, sin.train_Y, y_pred=y_pred)



if __name__ == "__main__":
    part_3_2()
    compare_to_ann()
