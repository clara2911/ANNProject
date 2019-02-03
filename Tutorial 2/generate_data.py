#!/usr/bin/env python
"""
File for all functions which read or generate data

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

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
