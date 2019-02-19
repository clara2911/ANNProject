#!/usr/bin/env python
"""
Main file for part 3.4

Authors: Kostis SZ, Romina Arriaza and Clara Tump
"""


import numpy as np
import data
from plot import show_tested
from plot import plot_accuracy
from hopfield_net import HopfieldNet


batch_epochs = 1
show_plots = True


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
    test_pattern = 2  # Choose between 0, 1, 2

    p_test = [p[test_pattern].flatten()]

    test_data = np.asarray(p_test)

    # Set noise percentages to test on [start, end, step]
    noise_percentages = np.arange(0, 101, 1)

    n_runs = 1
    runs = []
    for run in range(n_runs):
        acc = {}
        # Test for different percentages of noise
        for noise_perc in noise_percentages:
            # add noise to test data
            noisy_test_data = add_noise(test_data[0], noise_perc)
            # try to recall
            test_pred = h_net.recall([noisy_test_data], epochs=batch_epochs)

            acc[noise_perc] = calc_acc(test_data[0], test_pred[0])

            if show_plots:
                test_pred_1 = test_pred[0].reshape(32, 32)  # prepare for plotting

                show_tested(noisy_test_data, test_pred_1, test_pred_1.shape[0], test_pred_1.shape[1],
                            title="Testing with " + str(noise_perc) + "% noise")

        # plot_accuracy(acc)
        runs.append(acc)

    average_acc = {}
    for noise_perc in acc.keys():
        av_acc_i = 0.
        for run in range(n_runs):
            av_acc_i += runs[run][noise_perc]

        average_acc[noise_perc] = av_acc_i / float(n_runs)
    plot_accuracy(average_acc)


def add_noise(pattern, noise_percentage=0):
    """
    Add a specific amount of noise to the data.
    By noise we defined selecting a specific number of units and flipping them
    :param pattern: pattern to add noise to
    :param noise_percentage: how much noise to add
    :return the new noisy data
    """
    indices = range(pattern.shape[0])
    n_units_to_flip = int(pattern.shape[0] * (noise_percentage / 100.))
    picks = np.random.choice(indices, n_units_to_flip, replace=False)

    noisy_pattern = np.copy(pattern)

    for i in picks:
        noisy_pattern[i] = pattern[i] * -1

    return noisy_pattern


def calc_acc(original_pattern, predicted_pattern):
    """
    Calculate the accuracy of the model as the difference between the patterns.
    The flipped pattern also counts as a correct prediction.
    :param original_pattern: the target pattern
    :param predicted_pattern: the outcome of the model
    :return: accuracy: [0, 100]
    """
    acc = np.sum(original_pattern == predicted_pattern) / float(original_pattern.shape[0])
    negative_pattern = original_pattern * -1
    neg_acc = np.sum(negative_pattern == predicted_pattern) / float(original_pattern.shape[0])

    if neg_acc > acc:
        acc = - neg_acc
    return acc


if __name__ == '__main__':
    p = data.load_file()
    part_3_4()
