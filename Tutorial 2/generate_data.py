#!/usr/bin/env python
"""
File for all functions which read or generate data

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
import math


def sin_square(verbose=False):
    """
    Generate training and test data of a sin(2x) and a square(2x) function
    as indicated in the assignment's description.
    """
    # Data variables
    train_start = 0
    test_start = 0.05
    end = 2 * math.pi
    step = 0.1

    train_x = np.arange(train_start, end, step)
    test_x = np.arange(test_start, end, step)

    train_y = []
    test_y = []

    class sin(object):
        for x in train_x:
            train_y.append(math.sin(2*x))
        for x in test_x:
            test_y.append(math.sin(2*x))

        train_X = train_x
        train_Y = np.asarray(train_y)
        test_X = test_x
        test_Y = np.asarray(test_y)

    class square(object):
        train_X = train_x
        train_Y = np.where(sin.train_Y >= 0, 1, -1)
        test_X = test_x
        test_Y = np.where(sin.test_Y >= 0, 1, -1)

    return sin(), square()


def animal_data(verbose=False):
    """
    Reads in data files for assignment 4.1 Animal Ordering
    returns:
    - animal_feats: a matrix with features per animal
    - animal_names: an array of names of each animal (in same order as feats)
    """
    # animal_feats = np.fromfile('data/animals.dat', dtype=float)
    # animal_names = np.fromfile('data/animalnames.txt', dtype=float)
    animal_feats = np.array([[2,3,8],[4,5,9]])
    animal_names = np.array(['elephant','giraffe', 'cat'])
    if verbose:
        print("shape animal data: ", animal_feats.shape)
        print("shape animal names: ", animal_names.shape)
    return animal_feats, animal_names
