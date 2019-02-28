#!/usr/bin/env python
"""
Main file for Tutorial 4 Part 3.2

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import keras.backend as K
import data
import plot
from dnn import DNN

num_of_classes = 10


def main():

    params = {
        "lr": 0.15,
        "decay": 1e-6,
        "momentum": 0.1,
        "h_act_function": "sigmoid"
    }

    x_train, y_train, x_test, y_test = data.mnist(one_hot=True)

    # Define Deep Neural Network structure
    layers = [
        [x_train.shape[1], 512],
        [512, 256],
        [256, 128]
    ]

    # Initialize a deep neural network
    dnn = DNN(params)

    pre_epochs = 50
    train_epochs = 50

    # Create auto-encoders and train them one by one by stacking them in the DNN
    pre_trained_weights = dnn.pre_train(layers, x_train, pre_epochs)

    # Then use the pre-trained weights of these layers as initial weight values for the MLP
    history = dnn.train(x_train, y_train, train_epochs, init_weights=pre_trained_weights)
    # history = dnn.train(x_train, y_train, train_epochs)

    plot.plot_loss(history, loss_type='MSE')

    predicted, score = dnn.test(x_test, y_test)

    print("Test accuracy: ", score[1])


#def compare():


# def save_results():


if __name__ == "__main__":
    main()
