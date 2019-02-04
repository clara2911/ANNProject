#!/usr/bin/env python
"""
File for all functions which read or generate data

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

def animal_data(verbose=False, feat_lim=None, examples_lim=None):
    """
    Reads in data files for assignment 4.1 Animal Ordering
    returns:
    - animal_feats: a matrix with features per animal
    - animal_names: an array of names of each animal (in same order as feats)
    feat_lim/examples_lim: returns only the first M features or the
    first N examples (for debugging purposes)
    """
    animal_feats = np.loadtxt('data/animals.dat', dtype=int, delimiter=',')
    animal_names = np.loadtxt('data/animalnames.txt', dtype=str)
    num_examples = animal_names.shape[0]
    animal_feats = animal_feats.reshape(int(animal_feats.shape[0] / num_examples), num_examples)
    if feat_lim:
        animal_feats = animal_feats[:feat_lim, :]
    if examples_lim:
        animal_feats = animal_feats[:,:examples_lim]
        animal_names = animal_names[:examples_lim]
    if verbose:
        print("shape animal data: ", animal_feats.shape)
        print("shape animal names: ", animal_names.shape)
    return animal_feats, animal_names
