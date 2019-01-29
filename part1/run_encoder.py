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

    X_4 = np.array([[1, -1, 1, -1],
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        [1, -1, 1, -1]])


    X_8_triv = np.array([[1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1]])

    # Okay so the problem is that an 8x8 with randomly ordered elements is really difficult to model
    # if you input a very easy patter as shown above it can classify it correctly within 1 epoch
    # When you shuffle it is harder
    # When you have a majority of one class (for example the one_hot in generate data returns 54 of -1 and 10 of 1)
    # it will be biased to only predict one class
    # the above example of X is 50/50 so its easier

    np.random.shuffle([np.random.shuffle(X_row) for X_row in X_8_triv])

    X = X_8_triv
    print(X)
    params = {
        "learning_rate": 0.1,
        "batch_size": X.shape[1],  # setting it as 1 means sequential learning
        "theta": 0,
        "epsilon": 0.0,  # slack for error during training
        "epochs": 1000,
        "act_fun": 'step',
        "test_data": None,
        "test_targets": None,
        "m_weights": 0.1,
        "sigma_weights": 0.1,
        "beta": 1
    }

    # train ANN
    NN_structure = {
        0: 8,  # hidden layer
        1: X.shape[0]  # output layer
    }

    mlp = MLP(X, X, NN_structure, **params)
    out = mlp.train(verbose=verbose)
    print("predicted: ", out)
    print("targets: ", X)
    mlp.sum = out
    mlp.theta = 0.5
    out = mlp.step()
    print('predicted', out)
    correct, total = mlp.encoder_error(out, X)
    print('Correct {} out of {}'.format(correct, total))
    mlp.plot_error_history(mlp.error_history)

if __name__ == "__main__":
    main()
