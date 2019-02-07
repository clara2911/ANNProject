#!/usr/bin/env python
"""
Main file for assignment 3.3
RBF initialization using Competitive Learning (Vector Quantisation)

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""
import numpy as np
import generate_data
import plotter
from rbf_NN import RBF_Net
from vec_quant import VecQuantization

verbose = True


def part_3_3():
    """
    Regression problem on two functions (sin, square) using
    batch learning with least squares
    """
    method = 'least_squares'
    network_size = 10 # number of RBF nodes in the network # NOTE: larger than sample size
    sin, square = generate_data.sin_square(verbose=verbose)
    data = sin  # use which dataset to train and test

    rbf_net = RBF_Net(network_size, data.train_X)
    rbf_net2 = RBF_Net(network_size, data.train_X)

    vec_quant1 = VecQuantization(rbf_net2.RBF_Layer, iterations=1000, step_size=0.2, neighbor_bool = True)
    rbf_net2.RBF_Layer = vec_quant1.move_RBF(data.train_X)

    report_error(data, method, rbf_net, "normal")
    report_error(data, method, rbf_net2, "with VQ")




def report_error(data, method, rbf_net, label):
    y_train_pred, train_error = rbf_net.train(data.train_X, data.train_Y, method)
    y_pred, test_error = rbf_net.test(data.test_X, data.test_Y)
    #
    print("----", label, "----")
    print('Train error: ', train_error)
    print('Test error: ', test_error)
    # for rbf_node in rbf_net.RBF_Layer:
    #     print("mu: ", rbf_node.mu, "   /    sigma: ", rbf_node.sigma)
    #
    # plotter.plot_2d_function(data.train_X, data.train_Y, y_pred=y_train_pred, title='Train')
    # plotter.plot_2d_function(data.test_X, data.test_Y, y_pred=y_pred, title='Test')


if __name__ == "__main__":
    part_3_3()
