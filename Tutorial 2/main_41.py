#!/usr/bin/env python
"""
Main file for assignment 4.1
Topological Ordering of Animal Species

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""
import numpy as np
from som import Som
import generate_data


def main():
    animal_feats, animal_names = generate_data.animal_data(verbose=True, feat_lim=None, examples_lim=None)

    params = {
        "epochs" : 20,
        "step_size" : 0.2,
        "num_nodes": 100
    }
    som1 = Som(animal_feats, **params)
    som1.train()


if __name__ == "__main__":
    main()