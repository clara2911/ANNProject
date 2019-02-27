#!/usr/bin/env python
"""
Main file for Tutorial 4 Part 3.2

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import data
import plot
from dnn import DNN


def main():
    x_train, y_train, x_test, y_test = data.mnist()

    params = {
        "lr": 0.15,
        "decay": 1e-6,
        "momentum": 0.1,
        "h_act_function": "sigmoid"     # TODO: WATCHOUT
    }

    num_of_classes = 10

    # Define Deep Neural Network structure
    layers = [
        [x_train.shape[1], 512],
        [512, 256],
        [256, num_of_classes]]

    # Initialize a deep neural network
    dnn = DNN(params)

    pre_epochs = 1
    train_epochs = 50

    # TODO: Convert targets to one hot encoding

    # Create auto-encoders and train them one by one by stacking them in the DNN
    pre_trained_weights = dnn.pre_train(layers, x_train, pre_epochs)

    # Then use the pre-trained weights of these layers as initial weight values for the MLP
    history = dnn.train(pre_trained_weights, x_train, y_train, train_epochs)

    plot.plot_loss(history, loss_type='MSE')

    predicted = dnn.test(x_test)


if __name__ == "__main__":
    main()
