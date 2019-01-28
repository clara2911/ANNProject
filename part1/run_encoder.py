#!/usr/bin/env python
"""
Main file for data generalization with an auto-encoder

Authors: Clara Tump, Kostis SZ and Romina Azz
"""

import numpy as np
import matplotlib.pylab as plt
from generate_data import DataBase
from ann import ANN
from two_layer import MLP

def main():
    # Data variables
    N = 8
    data_base = DataBase()
    X = data_base.one_hot(N, 1, -1) #N, pos, neg
    encoder_learn(X)

def encoder_learn(X):
    verbose = False
    params = {
        "learning_rate": 0.1,
        "batch_size": X.shape[1],  # setting it as 1 means sequential learning
        "theta": 0,
        "epsilon": 0.0,  # slack for error during training
        "epochs": 100,
        "act_fun": 'step',
        "test_data": None,
        "test_targets": None,
        "m_weights": 0.1,
        "sigma_weights": 0.1,
        "beta": 1
    }

    # train ANN
    NN_structure = {
        0: 3,  # hidden layer
        1: X.shape[0]  # output layer
    }
    mlp = MLP(X, X, NN_structure, **params)
    out = mlp.train(verbose=verbose)
    print("predicted: ", out)
    print("targets: ", X)
    # mlp.plot_error_history()
    # mlp.test(test_X, test_Y)

if __name__ == "__main__":
    main()
