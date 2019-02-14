#!/usr/bin/env python
"""
Plotting functions

Authors: Kostis SZ, Romina Ariazza and Clara Tump

1: black
0: white
reshape order: left-right / top-bottom

So [1,1,-1,1] becomes

black black
white black
"""

import matplotlib.pyplot as plt
import math

def show_tested(pattern_in, pattern_out, dim1, dim2, title="Testing patterns"):
    """
    Input one input pattern and its output pattern
    Plots it as a picture
    """
    num_feats = pattern_in.shape[0]
    if dim1*dim2 == num_feats:
        plt.figure()
        ax1 = plt.subplot(121)
        ax1.set_title("input")
        pattern_in = pattern_in.reshape(dim1,dim2)
        ax1.imshow(pattern_in, cmap='binary')
        ax2 = plt.subplot(122)
        ax2.set_title("output")
        pattern_out = pattern_out.reshape(dim1, dim2)
        ax2.imshow(pattern_out, cmap='binary')
        plt.suptitle(title)
        plt.show()
    else:
        exit("EXITING PLOT: dim1*dim2 needs to be equal to num_feats")

def show_trained(train_set, dim1,dim2):
    """
    Plots the trained patterns
    """
    num_train = train_set.shape[0]
    num_feats = train_set.shape[1]
    if num_feats == dim1*dim2:
        plt.figure()
        for i in range(num_train):
            ax1 = plt.subplot(1,num_train,i+1)
            ax1.set_title("input pattern "+str(i+1))
            pattern_1 = train_set[i].reshape(dim1,dim2)
            ax1.imshow(pattern_1, cmap='binary')
        plt.suptitle("Trained patterns")
        plt.show()
    else:
        exit("EXITING PLOT: dim1*dim2 needs to be equal to num_feats")
