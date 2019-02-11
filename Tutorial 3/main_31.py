#!/usr/bin/env python
"""
Main file for assignment 3.1

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
import hopfield_net as hop_net
from plot import show_trained, show_tested


def main():
    """ main function"""

    x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
    train_set = np.vstack((x1, x2))
    train_set = np.vstack((train_set, x3))


    params = {
        "epochs": 100,
        "neurons": len(x1),
        "learn_method": 'classic'
    }

    hop = hop_net.HopfieldNet(train_set, **params)
    hop.batch_train()
    show_trained(train_set, 4,2)

    x1d = np.array([1, -1, 1 ,-1 ,1, -1, -1 , 1])
    x2d = np.array([1, 1, -1, -1 , -1, 1 , -1, -1])
    x3d = np.array([1, 1, 1, -1 ,1, 1 , -1 ,1])
    x12d = np.vstack((x1d, x2d))
    test_half = np.vstack((x12d, x3d))
    test_set = np.vstack((test_half, train_set))
    recalled_set = hop.recall(test_set)

    for i in range(test_set.shape[0]):
        show_tested(test_set[i], recalled_set[i], 4, 2)



if __name__ == '__main__':
    main()

