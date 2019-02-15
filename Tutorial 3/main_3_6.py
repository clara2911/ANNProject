#!/usr/bin/env python
"""
Main file for assignment 3.6

Authors: Kostis SZ, Romina Ariazza and Clara Tump

"""

import numpy as np
import itertools
from collections import defaultdict
import hopfield_net as hop_net
from plot import show_trained, show_tested, plot_capacity
from data import generate_sparse, distort

import time

def main():
    """
    main function
    Tests the capacity of the network for recalling exactly the train_set
    for different values for the bias
    """
    num_feats =  256 #512
    sparseness = 0.05 # 0.1, 0.01
    num_samples = [2,4,6,8,10,12,14,16,18,20,25,30]
    bias_list = [-1,-0.5,-0.1,0,0.1,0.5,1]
    num_dict = defaultdict(list)
    acc_dict = defaultdict(list)
    for bias in bias_list:
        print("bias: ", bias)
        for num in num_samples:
            print("num: ", num)
            train_set = generate_sparse(num, num_feats, sparseness)
            # test_set = distort(train_set, distortion)
            test_set = train_set
            recalled_set = train_and_test(train_set, test_set, bias, sparseness)
            accuracy = report_acc(recalled_set, train_set)

            num_dict[bias].append(num)
            acc_dict[bias].append(accuracy)

    plot_capacity(num_dict, acc_dict, bias_list, sparseness, num_feats)


def train_and_test(train_set, test_set, bias, sparseness):
    hop = hop_net.HopfieldNet(train_set)
    hop.sparse_train(sparseness)
    # batch or sequential
    # recalled_set = hop.recall_01s(test_set, epochs=100, threshold=bias)
    recalled_set = hop.sequential_recall_01s(test_set, epochs=8000, threshold=bias)
    return recalled_set

def report_acc(recalled_set, train_set):
    correct = 0
    for recalled_item in recalled_set:
        for trained_item in train_set:
            if np.array_equal(trained_item, recalled_item):
                correct += 1
                break
    acc =  correct / recalled_set.shape[0]
    return acc



if __name__ == '__main__':
    main()