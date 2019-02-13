#!/usr/bin/env python
"""
Main file for assignment 3.1

Authors: Kostis SZ, Romina Ariazza and Clara Tump

"""

# CHECK WHY -I makes sure diagonal = 0

import numpy as np
import itertools
import hopfield_net as hop_net
from plot import show_trained, show_tested


def main():
    """ main function"""

    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
    train_set = np.vstack((x1,x2, x3))

    hop = hop_net.HopfieldNet(train_set)
    hop.batch_train()
    show_trained(train_set, 4,2)

    x1d = np.array([1, -1, 1 ,-1 ,1, -1, -1 , 1])
    x2d = np.array([1, 1, -1, -1 , -1, 1 , -1, -1])
    x3d = np.array([1, 1, 1, -1 ,1, 1 , -1 ,1])
    test_set = np.vstack((train_set, x1d, x2d, x3d))
    recalled_set = hop.recall(test_set, epochs=100)

    for i in range(test_set.shape[0]):
        show_tested(test_set[i], recalled_set[i], 4, 2)

if __name__ == '__main__':
    main()