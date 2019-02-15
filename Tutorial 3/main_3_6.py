#!/usr/bin/env python
"""
Main file for assignment 3.6

Authors: Kostis SZ, Romina Ariazza and Clara Tump

"""

import numpy as np
import itertools
import hopfield_net as hop_net
from plot import show_trained, show_tested
from data import generate_sparse, distort

def main():
    """ main function"""
    num_feats = 1024
    sparseness = 0.1
    num_samples = [5, 10, 50, 100, 200]
    distortion = 0.1
    # which values to test for bias?
    bias = 0.
    for num in num_samples:
        train_set = generate_sparse(num, num_feats, sparseness)
        test_set = distort(train_set, distortion)
        accuracy = train_and_test(train_set, test_set, bias, sparseness)
        print(accuracy)


def train_and_test(train_set, test_set, bias, sparseness):
    hop = hop_net.HopfieldNet(train_set)
    hop.sparse_train(sparseness)
    # show_trained(train_set, 32, 32)

    # batch or sequential
    # recalled_set = hop.recall_01s(test_set, epochs=15, threshold=bias)
    recalled_set = hop.sequential_recall_01s(test_set, epochs=10, threshold=bias)
    correct = 0
    for recalled_item in recalled_set:
        for trained_item in train_set:
            if np.array_equal(trained_item, recalled_item):
                correct += 1
    acc =  correct / test_set.shape[0]
    return acc



if __name__ == '__main__':
    main()