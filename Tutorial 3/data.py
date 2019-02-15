#!/usr/bin/env python
"""
data.py
Functions to read contents of data file or generate data

Authors: Kostis SZ, Romina Arriaza and Clara Tump
"""

import numpy as np


def load_file():
    """
    Read file that has one line of integers separated by commas (,)
    :return: a dictionary where values are 2D arrays of 32x32
    """
    num_of_images = 11
    image_len = 1024
    dim_i_len = 32
    dim_j_len = 32

    p = {}

    with open('data/pict.dat') as f:
        data = [int(s) for s in f.readline().split(',')]

    for image in range(num_of_images):
        image_data = np.zeros((dim_i_len, dim_j_len))
        for dim_i in range(dim_i_len):
            for dim_j in range(dim_j_len):
                index = (image * image_len) + (dim_i * dim_i_len) + dim_j
                image_data[dim_i][dim_j] = data[index]
        p[image] = image_data

    return p

def generate_sparse(num_samples, num_feats, sparseness):
    """
    generate sparse patterns for assignment 3.6.
    Possible values are 0 and 1.
    Ratio of 1s to 0s is determined by the sparseness variable.
    """
    data = np.random.choice([0, 1], size=(num_samples,num_feats), p=[1-sparseness, sparseness])
    return data

def distort(data_set, distortion):
    distorted_set = np.copy(data_set)
    for i in range(distorted_set.shape[0]):
        num_flips = int(distortion * distorted_set.shape[1])
        idx = np.random.choice(distorted_set.shape[1], num_flips, replace=False)
        distorted_set[i,idx] = 1 - distorted_set[i,idx]
    return distorted_set


