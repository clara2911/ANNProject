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
    network_size = 100 # number of RBF nodes in the network #NOTE: larger than sample size

    sin, square = generate_data.sin_square(verbose=verbose)

    rbf_net = RBF_Net(network_size, sin.train_X)

    y_pred = rbf_net.train(sin.train_X, sin.train_Y, method)

    plotter.plot_2d_function(sin.train_X, sin.train_Y, y_pred=y_pred)


if __name__ == "__main__":
    part_3_1()
