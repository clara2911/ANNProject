#!/usr/bin/env python
"""
Main file for part 3.2

Authors: Kostis SZ, Romina Arriaza and Clara Tump
"""

import numpy as np
import data
from plot import show_trained, show_tested
from hopfield_net import HopfieldNet


batch_epochs = 1
seq_epochs = 6000

def part_3_2():
    """
    Compare batch vs sequential recall
    :param p: dictionary of different patterns
    """

    # Train with 3 patterns
    p_train = [p[0].flatten(),
               p[1].flatten(),
               p[2].flatten()]

    train_data = np.asarray(p_train)

    method = 'batch'
    #test_stability(train_data, method)
    #test_recovery(train_data, method)

    method = 'sequential'
    #test_stability(train_data, method)
    #test_recovery(train_data, method)


def test_stability(train_data, method):
    """
    Test that the patterns are stable
    :param train_data:
    :param method:
    :return:
    """
    h_net = HopfieldNet(train_data)

    h_net.batch_train()

    test_data = train_data

    if method == 'batch':
        print("Testing stability with batch")
        test_pred = h_net.recall(test_data, epochs=batch_epochs)
    else:
        print("Testing stability with sequential")
        test_pred, _ = h_net.sequential_recall(test_data, epochs=seq_epochs, plot_at_100=True)

    test_pred_1 = test_pred[0].reshape(32, 32)  # prepare for plotting
    test_pred_2 = test_pred[1].reshape(32, 32)  # prepare for plotting
    test_pred_3 = test_pred[2].reshape(32, 32)  # prepare for plotting

    show_tested(test_data[0], test_pred_1, test_pred_1.shape[0], test_pred_1.shape[1])
    show_tested(test_data[1], test_pred_2, test_pred_2.shape[0], test_pred_2.shape[1])
    show_tested(test_data[2], test_pred_3, test_pred_3.shape[0], test_pred_3.shape[1])


def test_recovery(train_data, method):
    """
    Test recovery abilities
    :param train_data:
    :param method:
    :return:
    """
    h_net = HopfieldNet(train_data)

    h_net.batch_train()

    p_test = [p[9].flatten(),
              p[10].flatten()]

    test_data = np.asarray(p_test)

    if method == 'batch':
        print("Testing recovery with batch")
        test_pred = h_net.recall(test_data, epochs=batch_epochs)
    else:
        print("Testing recovery with sequential")
        test_pred, _ = h_net.sequential_recall(test_data, epochs=seq_epochs, plot_at_100=True)

    test_pred_10 = test_pred[0].reshape(32, 32)  # prepare for plotting
    test_pred_11 = test_pred[1].reshape(32, 32)  # prepare for plotting

    show_tested(p_test[0], test_pred_10, test_pred_10.shape[0], test_pred_10.shape[1])
    show_tested(p_test[1], test_pred_11, test_pred_11.shape[0], test_pred_11.shape[1])


if __name__ == '__main__':
    p = data.load_file()
    part_3_2()
