#!/usr/bin/env python
"""
Test the test vectors for hopfield_net

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
import hopfield_net as hop_net
from plot import show_trained, show_tested


def main():
    """ Test for the hopfield network """
    """
    This is just for testing the functions
    """

    x1 = np.array([1, 1, 1, 1, -1, -1, 1, 1, 1])
    x2 = np.array([1, -1, 1, 1, 1, 1, 1, -1, 1])
    x3 = np.array([-1, 1, -1, -1, 1, -1, -1, 1, -1])
    train_set = np.vstack((x1, x2))
    train_set = np.vstack((train_set, x3))


    params = {
        "epochs": 100,
        "neurons": len(x1),
        "learn_method": 'classic'
    }

    hop = hop_net.HopfieldNet(train_set, **params)
    hop.batch_train()
    show_trained(train_set)

    x4d = [1,1,1,1,1,1,1,1,1]
    x5d = [1,1,1,1,-1,-1,1,-1,-1]
    x45d = np.vstack((x4d, x5d))
    test_set = np.vstack((x45d, train_set))
    recalled_set = hop.recall(test_set)
    for i in range(test_set.shape[0]):
        show_tested(test_set[i], recalled_set[i])



if __name__ == '__main__':
    main()

