#!/usr/bin/env python
"""
Main file for Tutorial 4 Part 3.1
Uses an auto-encoder (Autoencoder class) for different experiments

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
import data
import plot
from auto_encoder import Autoencoder

np.random.seed(10)

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
    auto_enc1 = Autoencoder(**params)
    history = auto_enc1.train(x_train, x_test)
    plot.plot_loss(history, loss_type='MSE')
    x_reconstr = auto_enc1.test(x_test, binary=True)
    plot_traintest(x_test, y_test, x_reconstr)

def plot_traintest(x_test, y_test, x_reconstr):
    true_ims_plot, reconstr_ims_plot = np.zeros([10, x_test.shape[1]]), np.zeros([10, x_reconstr.shape[1]])
    for i in range(10):
        indices, = np.where(y_test == i)
        index = indices[0]
        true_ims_plot[i] = x_test[index]
        reconstr_ims_plot[i] = x_reconstr[index]
    plot.plot_images(true_ims_plot, reconstr_ims_plot)

if __name__ == "__main__":
    main()