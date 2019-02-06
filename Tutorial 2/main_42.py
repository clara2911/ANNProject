#!/usr/bin/env python
"""
Main file for assignment 4.2
Cyclic Tour

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
    Reads in the city data, runs the som algorithm, and plots the topological 1-D mapping
    """
    city_coords, city_names = generate_data.city_coords(verbose=True, feat_lim=None, examples_lim=None)
    params = {
        "epochs" : 50,
        "step_size" : 0.2,
        "num_nodes": [10],
        "neighborhood_type": 'circular' # ['circular', 'linear']
    }
    som1 = Som(city_coords, **params)
    som1.train()
    pos = som1.apply(city_coords)

    sorted_coords, sorted_names = som1.order(pos, city_coords, city_names)
    print("city names in order: ")
    print(sorted_names)

    plot.plot_route(sorted_coords, sorted_names)

if __name__ == "__main__":
    main()