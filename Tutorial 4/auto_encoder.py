#!/usr/bin/env python
"""
contains class Autoencoder
Implements a single layer auto-encoder.
Functions:
 - train
 - test

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras import optimizers
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.initializers import RandomNormal


class Autoencoder:

    def __init__(self, **kwargs):
        """
        Initialize algorithm with data and parameters
        """
        var_defaults = {
            "bias_init": 'zeros',
            "weight_init": [0.0, 0.1],
            "seed": None,
            "num_hid_nodes": 32,
            "activations": 'sigmoid',
            "lr": 0.1,
            "decay": 0,
            "momentum": 0,
            "nesterov": False,
            "loss": 'mean_squared_error',
            "epochs": 10,
            "batch_size": 256,
            "verbose": 2
        }
        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.autoencoder = Sequential()

    def pre_train(self, layers, x_train, epochs):
        """
        Pretrain layer by layer the DNN
        :param layers: dictionary with layer and its number of hidden nodes
        :param x_train: training data
        :param epochs: how many epochs to train each layer
        """

        # greedy one by one layer pre-training
        for layer_i, nodes_i in layers.items():
            self.add_layer(nodes_i, x_train, epochs)

    def add_layer(self, num_of_nodes, x_train, epochs):
        """
        Add another layer to the model and train it
        :return:
        """
        # save output layer
        output_layer = self.autoencoder.layers[-1]

        # remove output layer
        self.autoencoder.pop()

        # mark all other layers as non trainable, meaning dont update them while training
        for layer in self.autoencoder.layers:
            layer.trainable = False

        # add a new hidden layer to train
        self.autoencoder.add(Dense(num_of_nodes, activation='relu',
                                   kernel_initializer='he_uniform'))

        # add the output layer back again
        self.autoencoder.add(output_layer)

        # pre-train the model
        self.autoencoder.fit(x_train, x_train, epochs=epochs, verbose=0)

    def train(self, x_train, x_val):
        """
        :param x_train: Training data
        :param x_val: Validation data
        :return:
        """
        dim = x_train.shape[1]

        rand_norm = RandomNormal(mean=self.weight_init[0], stddev=self.weight_init[1], seed=self.seed)

        self.autoencoder.add(Dense(self.num_hid_nodes,
                                   activation=self.activations,
                                   kernel_initializer=rand_norm,
                                   bias_initializer=self.bias_init,
                                   input_dim=dim))

        # last layer should always be sigmoid so that out is in [0,1]
        self.autoencoder.add(Dense(dim, activation='sigmoid'))

        sgd = optimizers.SGD(lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=self.nesterov)

        self.autoencoder.compile(loss=self.loss, optimizer=sgd)

        history = self.autoencoder.fit(x_train, x_train,
                                       epochs=self.epochs,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       verbose=self.verbose,
                                       validation_data=(x_val, x_val))
        return history

    def test(self, x_test, binary=True, batch_size=32, verbose=1):
        """
        :param x_test: Testing data
        :param binary: return output in a binary format
        :param batch_size: self-explanatory
        :param verbose: Show testing info
        """
        x_reconstr = self.autoencoder.predict(x_test, batch_size=batch_size, verbose=verbose)
        if binary:
            x_reconstr = np.where(x_reconstr < 0.5, 0, 1)
        return x_reconstr
