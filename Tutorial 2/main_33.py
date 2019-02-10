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
    method = 'delta_rule'#'least_squares'
    network_size = 50 # number of RBF nodes in the network # NOTE: larger than sample size
    sin, square = generate_data.sin_square(verbose=verbose)
    data = square  # use which dataset to train and test
    # file = np.loadtxt('data/ballist.dat', dtype=float)
    # data.train_X = file[:,[0,1]]
    # data.train_Y = file[:, [2, 3]]
    # file = np.loadtxt('data/balltest.dat', dtype=float)
    # data.test_X = file[:, [0, 1]]
    # data.test_Y = file[:, [2, 3]]

    #noisy_data = add_noise_to_data(data)
    #data = noisy_data

    rbf_net = RBF_Net(network_size, data.train_X)
    rbf_net2 = RBF_Net(network_size, data.train_X)

    vec_quant1 = VecQuantization(rbf_net2.RBF_Layer, iterations=10000, step_size=0.001, neighbor_bool = True)
    rbf_net2.RBF_Layer = vec_quant1.move_RBF(data.train_X)

    #report_error(data, method, rbf_net, "normal", lr = 0.001)
    report_error(data, method, rbf_net2, "with VQ", lr = 0.001)

    print('hola')

# ---- normal ----
# Train error:  [0.00785915]
# Test error:  [0.01412971]
# ---- with VQ ----
# Train error:  [0.00259548]
# Test error:  [0.62155973]



def report_error(data, method, rbf_net, label, lr = 0.001):
    y_train_pred, train_error = rbf_net.train(data.train_X, data.train_Y, method, lr=lr)
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

def add_noise_to_data(data):
    """
    Add noise to all the train and test data samples
    """
    data.train_X = data.train_X + np.random.normal(0, 0.01, data.train_X.shape)  # zero mean, 0.1 std
    data.train_Y = data.train_Y + np.random.normal(0, 0.01, data.train_Y.shape)
    data.test_X = data.test_X + np.random.normal(0, 0.01, data.test_X.shape)
    data.test_Y = data.test_Y + np.random.normal(0, 0.01, data.test_Y.shape)
    return data

if __name__ == "__main__":
    part_3_3()
