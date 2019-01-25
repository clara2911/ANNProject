#!/usr/bin/env python
"""
Main file for data generalization with an auto-encoder

Authors: Clara Tump, Kostis SZ... and Romina Azz...
"""

import numpy as np
import matplotlib.pylab as plt
from generate_data import DataBase
from ann import ANN

def main():
    # Data variables
    N = 4
    data_base = DataBase()
    X = data_base.one_hot(N)
    print(X)
    encoder_learn(X)

def encoder_learn(X):
    pass
    # # here we use batch
    # verbose = True
    # fig, ax = plt.subplots()
    # params = {
    #     "learning_rate": 0.1,
    #     "batch_size": 1,
    #     "theta": 0,
    #     "epsilon": 0.0,  # slack for error during training
    #     "epochs": 50,
    #     "act_fun": 'step',
    #     "test_data": None,
    #     "test_targets": None,
    #     "m_weights": 0,
    #     "sigma_weights": 0.5,
    #     "nodes": 3,
    #     "learn_method": 'perceptron',  # 'delta_rule' or 'perceptron'
    #     "bias": 0
    # }
    # ann = ANN(X, Y, **params)
    # ann.train_batch(verbose=verbose)
    # # ax.plot(range(len(ann.error_history)), ann.error_history)
    # ann.plot_decision_boundary(
    #         data=ann.train_data,
    #         plot_intermediate=True,
    #         title='Learning without bias',
    #         data_coloring = ann.train_targets,
    #         origin_grid=True
    #         )

if __name__ == "__main__":
    main()