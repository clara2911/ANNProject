#!/usr/bin/env python
"""
Main file for Tutorial 4 Part 3.1
Uses an auto-encoder (Autoencoder class) for different experiments

Authors: Kostis SZ, Romina Arriaza and Clara Tump
"""

import numpy as np
import data
import plot
from auto_encoder import Autoencoder

np.random.seed(10)

def main():
    x_train, y_train, x_test, y_test = data.mnist()
    layer_sizes = [10,100,400,784,900,1000]
    sigmoid_histories = []
    relu_histories = []
    for layer_size in layer_sizes:
        print("layer_size: ", layer_size)
        sigm_1run = train(layer_size, 'sigmoid', x_train, x_test)
        relu_1run = train(layer_size, 'relu', x_train, x_test)
        sigmoid_histories.append(sigm_1run)
        relu_histories.append(relu_1run)
        print("MSE sigmoid: ", sigm_1run)
        print("MSE relu: ", relu_1run)
        print("sigmoid histories: ", sigmoid_histories)
        print("relu histories: ", relu_histories)
    plot.plot_losses(layer_sizes, sigmoid_histories, relu_histories)

def train(layer_size, activation, x_train, x_test):
    params = {
            "epochs" : 35,
            "num_hid_nodes": layer_size,
            "weight_init": [0.0, 0.1],
            "activations": activation, #relu is much better performance
            "lr": 0.15,
            "verbose": 0
        }
    auto_enc1 = Autoencoder(**params)
    if layer_size > 784:
        regularization = 0.0001
    else:
        regularization = 0
    history = auto_enc1.train(x_train,x_train, x_test, regularization=regularization)
    # loss = auto_enc1.evaluate(x_test)
    # print("history: ", history.history['loss'])
    return history.history['loss']


if __name__ == "__main__":
    main()