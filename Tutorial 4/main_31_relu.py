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
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_test = x_test[:5000]
    y_test = y_test[:5000]
    lrs = [0.01,0.1,1,10,15,20,50]
    errors = []
    stds = []
    for lr in lrs:
        print("learning rate: ", lr)
        single_errors = []
        for i in range(3):
            error_1run = train(lr, x_train, x_test, y_test)
            single_errors.append(error_1run)
        errors.append(np.mean(single_errors))
        stds.append(np.std(single_errors))
        print("MSE: ", np.mean(single_errors), " // std: ", np.std(single_errors))
    plot.plot_parameter('Learning rate', lrs, errors, stds)

def train(lr, x_train, x_test, y_test):
    params = {
            "epochs" : 50,
            "num_hid_nodes": int(x_train.shape[1]*0.7),
            "weight_init": [0.0, 0.1],
            "activations": 'relu', #relu is much better performance
            "lr": lr,
            "verbose": 0
        }
    auto_enc1 = Autoencoder(**params)
    history = auto_enc1.train(x_train,x_train, x_test)

    # plot.plot_loss(history, loss_type='MSE')
    # x_reconstr = auto_enc1.test(x_test, binary=True)
    # plot_traintest(x_test, y_test, x_reconstr)
    return history.history['loss'][-1]

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