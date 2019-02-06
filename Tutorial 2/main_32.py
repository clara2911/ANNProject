#!/usr/bin/env python
"""
Main file for assignment 3.2
RBF Network on a sin and square function

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""
import numpy as np
import generate_data
from ann import ANN

verbose = True


def part_3_2():
    """
    Regression problem on two functions (sin, square) with added noise
    using online (sequential) learning with delta rule
    """
    method = 'delta_rule'
    network_size = 50

    sin, square = generate_data.sin_square(verbose=verbose)
    sin = add_noise_to_data(sin)
    square = add_noise_to_data(square)

    rbf_net = RBF_Net(network_size)

    rbf_net.train(sin.train_X, sin.train_Y, method)


def add_noise_to_data(data):
    data.train_sin_x = generate_data.add_noise(data.train_sin_x)
    data.train_sin_y = generate_data.add_noise(data.train_sin_y)
    data.test_sin_x = generate_data.add_noise(data.test_sin_x)
    data.test_sin_y = generate_data.add_noise(data.test_sin_y)
    return data


def compare_to_ann():
    """
    Regression problem on two functions (sin, square) with added noise
    using batch learning with a 2 layer neural network
    """
    method = 'delta_rule'
    network_size = 50

    sin, square = generate_data.sin_square(verbose=verbose)
    sin = add_noise_to_data(sin)
    square = add_noise_to_data(square)

    ann = ANN()

    ann.solve(sin.train_X, sin.train_Y, sin.test_X, sin.test_Y)


if __name__ == "__main__":
    part_3_2()
    compare_to_ann()
