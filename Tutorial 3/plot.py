#!/usr/bin/env python
"""
Test the test vectors for hopfield_net

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import matplotlib.pyplot as plt
import math

def show_tested(pattern_in, pattern_out):
    """
    Input one input pattern and its output pattern
    Plots it as a picture
    """
    num_feat = pattern_in.shape[0]
    if math.sqrt(num_feat) * math.sqrt(num_feat) == num_feat:
        plt.figure()
        ax1 = plt.subplot(121)
        ax1.set_title("input")
        pattern_in = pattern_in.reshape(int(math.sqrt(num_feat)),int(math.sqrt(num_feat)))
        ax1.imshow(pattern_in, cmap='binary')
        ax2 = plt.subplot(122)
        ax2.set_title("output")
        pattern_out = pattern_out.reshape((3, 3))
        ax2.imshow(pattern_out, cmap='binary')
        plt.suptitle("Testing patterns")
        plt.show()
    else:
        exit("EXITING PLOT: number of features should be square (3x3) or (2x2) etc")

def show_trained(train_set):
    """
    Plots the trained patterns
    Only works if the length of the features is square (3x3 or 2x2
    """
    num_train = train_set.shape[0]
    num_feat = train_set.shape[1]
    if math.sqrt(num_feat)*math.sqrt(num_feat) == num_feat:
        plt.figure()
        for i in range(num_train):
            ax1 = plt.subplot(1,num_train,i+1)
            ax1.set_title("input pattern "+str(i+1))
            pattern_1 = train_set[i].reshape((int(math.sqrt(num_feat)),int(math.sqrt(num_feat))))
            ax1.imshow(pattern_1, cmap='binary')
        plt.suptitle("Trained patterns")
        plt.show()
    else:
        exit("EXITING PLOT: number of features should be square (3x3) or (2x2) etc")
