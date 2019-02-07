#!/usr/bin/env python
"""
Wrapper file for plotting function

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

from matplotlib import pyplot as plt


def plot_2d_function(x, y, y_pred=None, title=None):
    plt.scatter(x, y, color='blue')

    if y_pred is not None:
        plt.scatter(x, y_pred, color='red')

    plt.title(title)
    plt.show()


def plot_2d_function_multiple(x, y, y_pred, title=None):
    plt.plot(x, y, color='blue')

    for y_p in y_pred:
        plt.plot(x, y_p)

    plt.title(title)
    plt.show()


def plot_errors(x, y, title=None):
    plt.figure()
    plt.errorbar(x, y, fmt='o')
    labels = [str(i) for i in x]
    plt.xticks(x, labels)
    plt.title(title)
    plt.show()
