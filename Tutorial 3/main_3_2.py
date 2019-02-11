#!/usr/bin/env python
"""
Main file for part 3.2

Authors: Kostis SZ, Romina Arriaza and Clara Tump
"""

import numpy as np
import data
from plot import show_trained, show_tested
from hopfield_net import HopfieldNet


def part_3_2(p):
    """

    :param p:
    :return:
    """

    test_stability = False
    test_recovery = False

    # First train with 3 patterns
    p_train = [p[0].flatten(),
               p[1].flatten(),
               p[2].flatten()]

    train_data = np.asarray(p_train)

    h_net = HopfieldNet(train_data)

    h_net.batch_train()

    # Q1: Test that the patterns are stable
    if test_stability:
        p_test = p_train

        test_data = np.asarray(p_test)

        test_pred = h_net.recall(test_data, epochs=500)

        test_pred_1 = test_pred[0].reshape(32, 32)  # prepare for plotting
        test_pred_2 = test_pred[1].reshape(32, 32)  # prepare for plotting
        test_pred_3 = test_pred[2].reshape(32, 32)  # prepare for plotting

        show_tested(p_test[0], test_pred_1, test_pred_1.shape[0], test_pred_1.shape[1])
        show_tested(p_test[1], test_pred_2, test_pred_2.shape[0], test_pred_2.shape[1])
        show_tested(p_test[2], test_pred_3, test_pred_3.shape[0], test_pred_3.shape[1])

    # Q2: Test recovery abilities
    if test_recovery:
        p_test = [p[9].flatten(),
                  p[10].flatten()]

        test_data = np.asarray(p_test)

        test_pred = h_net.recall(test_data, epochs=500)

        test_pred_10 = test_pred[0].reshape(32, 32)  # prepare for plotting
        test_pred_11 = test_pred[1].reshape(32, 32)  # prepare for plotting

        show_tested(p_test[0], test_pred_10, test_pred_10.shape[0], test_pred_10.shape[1])
        show_tested(p_test[1], test_pred_11, test_pred_11.shape[0], test_pred_11.shape[1])

    # Q3: Sequential update





if __name__ == '__main__':
    p = data.load_file()
    part_3_2(p)