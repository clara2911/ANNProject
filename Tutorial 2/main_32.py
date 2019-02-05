#!/usr/bin/env python
"""
Main file for assignment 3.2
RBF Network on a sin and square function

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""
import numpy as np
import generate_data
from rbf_NN import RBF_Net

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


if __name__ == "__main__":
    part_3_2()
