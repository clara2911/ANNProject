#!/usr/bin/env python
"""
Main file for Tutorial 4 Part 3.2

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
import data
import plot
from dnn import DNN


def main():
    x_train, y_train, x_test, y_test = data.mnist()

    params = {
            "epochs" : 50,
            "num_hid_nodes": int(x_train.shape[1]*0.9),
            "weight_init": [0.0, 0.1],
            "activations": 'sigmoid', #relu is much better performance
            "lr": 0.15,
            "decay": 1e-6,
            "momentum": 0.1,
        }

    dnn = DNN()

    dnn.pre_train()

    history = dnn.train()

    plot.plot_loss(history, loss_type='MSE')

    predicted = dnn.test()


if __name__ == "__main__":
    main()
