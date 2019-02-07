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
    animal_feats = np.transpose(animal_feats.reshape(num_examples, int(animal_feats.shape[0] / num_examples)))
    if feat_lim:
        if type(feat_lim) == int:
            animal_feats = animal_feats[:feat_lim, :]
        elif isinstance(feat_lim, list):
            animal_feats = animal_feats[feat_lim]
    if examples_lim:
        if type(examples_lim) == int:
            animal_feats = animal_feats[:,:examples_lim]
            animal_names = animal_names[:examples_lim]
        elif isinstance(examples_lim, list):
            animal_feats = animal_feats[:, examples_lim]
            animal_names = animal_names[examples_lim]
    if verbose:
        print("shape animal data: ", animal_feats.shape)
        print("shape animal names: ", animal_names.shape)
    return animal_feats, animal_names

def city_coords(verbose=False, feat_lim = None, examples_lim = None):
    city_coords = np.loadtxt('data/cities.dat', dtype=float, delimiter=',')
    city_names = np.loadtxt('data/city_names.txt', dtype=str)
    city_coords = np.transpose(city_coords)
    if feat_lim:
        if type(feat_lim) == int:
            city_coords = city_coords[:feat_lim, :]
        elif isinstance(feat_lim, list):
            city_coords = city_coords[feat_lim]
    if examples_lim:
        if type(examples_lim) == int:
            city_coords = city_coords[:,:examples_lim]
            city_names = city_names[:examples_lim]
        elif isinstance(examples_lim, list):
            city_coords = city_coords[:, examples_lim]
            city_names = city_names[examples_lim]
    if verbose:
        print("city_names: ", city_names)
        print("city_coords shape: ", city_coords.shape)
        print("city coords: ")
        print(city_coords)
    return city_coords, city_names
