#!/usr/bin/env python
"""
data.py
Function to read contents of data file

Authors: Kostis SZ, Romina Arriaza and Clara Tump
"""

import numpy as np


def load_file():
    """
    Read file that has one line of integers separated by commas (,)
    :return: a dictionary where values are 2D arrays of 32x32
    """
    num_of_images = 12
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
