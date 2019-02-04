#!/usr/bin/env python
"""
Main file for assignment 4.1
Topological Ordering of Animal Species

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""
import numpy as np
from som import Som
import generate_data
import plot
import random

random.seed(54)

def main():
    """
    Reads in the animal data, runs the som algorithm, and plots the topological 1-D mapping
    """
    # You can do experiments for only certain features / samples
    # Eg put feat_lim=[2,29] to only use 'barks' and 'flying' or set examples_lim=[0] to only consider the antelope
    animal_feats, animal_names = generate_data.animal_data(verbose=True, feat_lim=None, examples_lim=None)
    params = {
        "epochs" : 20,
        "step_size" : 0.2,
        "num_nodes": 100
    }
    som1 = Som(animal_feats, **params)
    som1.train()
    pos = som1.apply(animal_feats)
    sorted_feats, sorted_names = som1.order(pos, animal_feats, animal_names)
    print("animal names in order: ")
    print(sorted_names)
    plot.plot_order_1d(pos, animal_names)

if __name__ == "__main__":
    main()