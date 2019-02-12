#!/usr/bin/env python
"""
Main file for part 3.4

Authors: Kostis SZ, Romina Arriaza and Clara Tump
"""


import numpy as np
import data
from plot import show_trained, show_tested
from hopfield_net import HopfieldNet


batch_epochs = 500


def part_3_4():
    """
    Test hopfield network's robustness on different settings of noisy data
    :param p: dictionary of different patterns
    """

    # Train with 3 patterns
    p_train = [p[0].flatten(),
               p[1].flatten(),
               p[2].flatten()]

    train_data = np.asarray(p_train)

    h_net = HopfieldNet(train_data)

    h_net.batch_train()

    # Choose a pattern and add noise to it
    test_pattern = 0  # Choose between 0, 1, 2

    p_test = [p[test_pattern].flatten()]

    test_data = np.asarray(p_test)

    # Set noise percentages to test on [start, end, step]
    noise_percentages = np.arange(0, 1., 0.01)

    # Test for different percentages of noise
    for noise_perc in noise_percentages:
        # add noise to test data
        test_data[0] = add_noise(test_data[0], noise_perc)
        # try to recall
        test_pred = h_net.recall(test_data, epochs=batch_epochs)

        test_pred_1 = test_pred[0].reshape(32, 32)  # prepare for plotting

        show_tested(test_data[0], test_pred_1, test_pred_1.shape[0], test_pred_1.shape[1])


def add_noise(pattern, noise_percentage=0.1):
    """
    Add a specific amount of noise to the data.
    By noise we defined selecting a specific number of units and flipping them
    :param pattern: pattern to add noise to
    :param noise_percentage: how much noise to add
    :return the new noisy data
    """
    indices = range(pattern.shape[0])
    rand_pick = np.random.choice(indices, noise_percentage * 100, replace=False)



if __name__ == '__main__':
    p = data.load_file()
    part_3_4()
