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
    method = 'delta_rule'  # delta_rule , least_squares
    network_size = 63
    learning_rate = 0.001  # optimal value: 0.01 bigger overshoots, smaller underperforms

    sin, square = generate_data.sin_square(verbose=verbose)

    data = sin  # use which dataset to train and test
    noisy_data = add_noise_to_data(data)
    data = noisy_data

    random_mu = False

    iterations = 1
    iter_results = {}
    errors = []
    y_preds = []
    # number of RBF nodes in the network
    network_size_list = [63]# [2, 5, 10, 20, 30, 40, 50, 63] # NOTE: larger than sample size
    sigmas = [0.25] # , 0.5, 0.6, 0.75, 0.8, 0.9, 1.]
    #sigmas = [0.001, 0.1, 1., 10.]
    for i in range(iterations):
        for sigma in sigmas:
            for network_size in network_size_list:

                rbf_net = RBF_Net(network_size, data.train_X, random_mu=random_mu, sigma=sigma)

                y_train_pred, train_error = rbf_net.train(data.train_X, data.train_Y, method, lr=learning_rate)

                y_pred, test_error = rbf_net.test(data.test_X, data.test_Y)

                print('# Nodes: ', network_size, ' sigma: ', sigma)
                print('Train error: ', train_error)
                print('Test error: ', test_error)
                errors.append(test_error)
                y_preds.append(y_pred)
                # plotter.plot_2d_function(data.train_X, data.train_Y, y_pred=y_train_pred, title='Train')
                plotter.plot_2d_function(data.test_X, data.test_Y, y_pred=y_pred, title='Test')

            iter_results[i] = [errors, y_preds]
            #plotter.plot_2d_function(data.train_X, data.train_Y, y_pred=y_train_pred, title='Train')
            #plotter.plot_2d_function(data.test_X, data.test_Y, y_pred=y_pred, title='Test')

        #plotter.plot_errors(network_size_list, errors, title='Test error for delta rule')
        # print(rbf_net.weights[:20])

    # plotter.plot_2d_function_multiple(data.test_X, data.test_Y, y_preds)

    # plotter.plot_errors(sigmas, errors, title='Test error on different widths of nodes')

    #plotter.plot_2d_function(data.train_X, data.train_Y, y_pred=y_train_pred, title='Train')
    #plotter.plot_2d_function(data.test_X, data.test_Y, y_pred=y_pred, title='Test')


def train_noisy_test_clean():
    """
    Regression problem on two functions (sin, square) trained on noisy data
    but tested on clean data
    """
    method = 'least_squares'  # delta_rule , least_squares
    learning_rate = 0.001  # optimal value: 0.01 bigger overshoots, smaller underperforms

    sin, square = generate_data.sin_square(verbose=verbose)
    clean_data, _ = generate_data.sin_square(verbose=verbose)

    noisy_data = add_noise_to_data(sin)

    random_mu = False

    network_size = 63
    sigma = 0.5

    # TRAIN ON NOISY DATA
    rbf_net = RBF_Net(network_size, noisy_data.train_X, random_mu=random_mu, sigma=sigma)
    # TRAIN ON NOISY DATA
    y_train_pred, train_error = rbf_net.train(noisy_data.train_X, noisy_data.train_Y, method, lr=learning_rate)
    # TEST ON CLEAN DATA
    y_pred, test_error = rbf_net.test(noisy_data.test_X, noisy_data.test_Y)

    print('# Nodes: ', network_size, ' sigma: ', sigma)
    print('Train error: ', train_error)
    print('Test error: ', test_error)

    # plotter.plot_errors(sigmas, errors, title='Test error')

    plotter.plot_2d_function(data.train_X, data.train_Y, y_pred=y_train_pred, title='Train')
    plotter.plot_2d_function(noisy_data.test_X, noisy_data.test_Y, y_pred=y_pred, title='Test')


def compare_to_ann():
    """
    Regression problem on two functions (sin, square) with added noise
    using batch learning with a 2 layer neural network
    """

    epochs = 5000
    hidden_neurons = 63  # same number as RBF nodes
    output_neurons = 1

    sin, square = generate_data.sin_square(verbose=verbose)
    sin = add_noise_to_data(sin)
    square = add_noise_to_data(square)

    data = square  # use which dataset to train and test

    batch_size = data.train_X.shape[0]  # set batch size equal to number of data points

    ann = ANN(epochs, batch_size, hidden_neurons, output_neurons)

    y_pred = ann.solve(data.train_X, data.train_Y, data.test_X, data.test_Y)

    error = 0.
    for i in range(data.test_Y.shape[0]):
        error += np.abs(data.test_Y[i] - y_pred[i])

    test_error = error / data.test_Y.shape[0]

    print('Test error: ', test_error)

    plotter.plot_2d_function(data.test_X, data.test_Y, y_pred=y_pred)


def add_noise_to_data(data):
    """
    Add noise to all the train and test data samples
    """
    data.train_X = data.train_X + np.random.normal(0., 0.1, data.train_X.shape)  # zero mean, 0.1 std
    data.train_Y = data.train_Y + np.random.normal(0., 0.1, data.train_Y.shape)
    data.test_X = data.test_X + np.random.normal(0., 0.1, data.test_X.shape)
    data.test_Y = data.test_Y + np.random.normal(0., 0.1, data.test_Y.shape)
    return data


if __name__ == "__main__":
    # part_3_2()
    train_noisy_test_clean()
    # compare_to_ann()
