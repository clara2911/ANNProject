#!/usr/bin/env python
"""
A Deep Neural Network

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras import optimizers
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.initializers import RandomNormal


class DNN:

    def __init__(self, **kwargs):
        """
        Initialize algorithm with data and parameters
        """
        var_defaults = {
            "batch_size": 256,
            "act_function": 'sigmoid',
            "lr": 0.1,
            "decay": 0,
            "momentum": 0,
            "nesterov": False,
            "loss": 'mean_squared_error'
        }
        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.dnn = Sequential()

    def _setup_model(self, x_train, output_nodes):
        """
        Setup model parameters
        """
        input_dim = x_train.shape[1]

        # TODO: how many nodes?
        X = 32
        # add just one layer
        self.dnn.add(Dense(X, input_dim=input_dim, activation=self.act_function))
        # and then output layer
        self.dnn.add(Dense(output_nodes, activation='sigmoid'))

        sgd = optimizers.SGD(lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=self.nesterov)

        self.dnn.compile(loss=self.loss, optimizer=sgd)

    def pre_train(self, layers, x_train, pre_epochs):
        """
        Pretrain layer by layer the DNN
        :param layers: dictionary with layer and its number of hidden nodes
        :param x_train: training data
        :param pre_epochs: how many epochs to train each layer
        """

        output_nodes = layers[-1]
        # Initialize to a one layer NN
        self._setup_model(x_train, output_nodes)

        # greedy one by one layer pre-training
        for layer_i, nodes_i in layers.items():
            self.add_layer(nodes_i, x_train, pre_epochs)
            # here you can evaluate its performance when adding a new layer

    def add_layer(self, num_of_nodes, x_train, epochs):
        """
        Add another layer to the model and train it
        :return:
        """
        # save output layer
        output_layer = self.dnn.layers[-1]

        # remove output layer
        self.dnn.pop()

        # mark all other layers as non trainable, meaning dont update them while training
        for layer in self.dnn.layers:
            layer.trainable = False

        # TODO: what kernel initializer should we use?
        # - he_uniform
        # random normal

        # add a new hidden layer to train
        self.dnn.add(Dense(num_of_nodes, activation=self.act_function,
                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.2)))

        # add the output layer back again
        self.dnn.add(output_layer)

        # pre-train the model
        self.dnn.fit(x_train, x_train, epochs=epochs, verbose=0)

    def train(self, x_train, x_val):
        """
        :param x_train: Training data
        :param x_val: Validation data
        :return:
        """


        history = self.dnn.fit(x_train, x_train,
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
