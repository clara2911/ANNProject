#!/usr/bin/env python
"""
File for all functions which make plots

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_ordering_1d(pos, animal_names):
    """
    For assignment 4.1
    plots the animal_names on a 1-D line ordered by a topological ordering supplied by the array pos
    """
    fig, ax = plt.subplots(1)
    ax.scatter(np.zeros(len(pos)), pos, s=1, color='purple')
    for i, name in enumerate(animal_names):
        ax.annotate(name, (0, pos[i]), rotation=0, fontsize=8, color='blue')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title("Topological Ordering of Animal Species")
    plt.show()

def plot_ordering_2d(pos, animal_names):
    """
    For assignment 4.1
    plots the animal_names on a 2-D grid ordered by a topological ordering supplied by the array pos
    """
    fig, ax = plt.subplots(1)
    ax.scatter(pos[:,0], pos[:,1], s=1, color='purple')
    for i, name in enumerate(animal_names):
        ax.annotate(name, (pos[i][0], pos[i][1]), rotation=45, fontsize=8, color='blue')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title("Topological Ordering of Animal Species")
    plt.show()
    pass