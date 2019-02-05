#!/usr/bin/env python
"""
Wrapper file for plotting function

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

from matplotlib import pyplot as plt


def plot_2d_function(x, y, y_pred=None):
    plt.plot(x, y, color='blue')

    if y_pred is not None:
        plt.plot(x, y_pred, color='red')

    plt.show()
