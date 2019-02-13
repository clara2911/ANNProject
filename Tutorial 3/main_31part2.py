#!/usr/bin/env python
"""
Main file for assignment 3.1, the second question (about the number of attractors)

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

    train_set = np.vstack((x1, x2, x3))

    hop = hop_net.HopfieldNet(train_set)
    hop.batch_train()
    # all_poss is all possible 8-dimensional input vectors (256 vectors)
    all_poss = np.array(list(itertools.product([-1, 1], repeat=8)))
    recalled_set = hop.recall(all_poss, epochs=10)
    print("recalled_set shape: ", recalled_set.shape)
    unique_info = np.unique(recalled_set, return_index=True, return_inverse=True, return_counts=True, axis=0)
    print("num attractors: ", unique_info[0].shape)

if __name__ == '__main__':
    main()

