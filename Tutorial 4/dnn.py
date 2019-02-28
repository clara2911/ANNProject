#!/usr/bin/env python
"""
A Deep Neural Network

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping


class DNN:

    def __init__(self, kwargs):
        """
        Initialize algorithm with data and parameters
        """
        var_defaults = {
            "batch_size": 256,
            "h_act_function": 'sigmoid',
            "out_act_function": 'sigmoid',
            "lr": 0.1,
            "decay": 0,
            "momentum": 0,
            "loss": 'mean_squared_error'
        }
        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.model = None

    def pre_train(self, layers, x_train, pre_epochs):
        """
        Pre-train layer by layer the DNN
        :param layers: dictionary with layer and its number of hidden nodes
        :param x_train: training data
        :param pre_epochs: how many epochs to train each layer
        """
        # For the first layer the representation of the data is the same
        data_represent_i = x_train

        # A dictionary to save the weights of the encoders
        pre_trained_weights = {}

        for i, layer in enumerate(layers):
            # Input dimensions of next layer
            input_dim_of_layer = layer[0]

            # Number of nodes of the layer
            nodes_of_layer = layer[1]

            # Output dimensions of layer
            output_dim_of_layer = input_dim_of_layer

            # Define the input of an encoder which is always an image
            encoder_input = Input(shape=(input_dim_of_layer,))

            # Define a layer to transform the data in another representation
            encoder = Dense(nodes_of_layer, activation=self.h_act_function)(encoder_input)

            # For the autoencoder define an output layer to reconstruct the image
            decoder = Dense(output_dim_of_layer, activation=self.h_act_function)(encoder)

            # Define a simple autoencoder that receives a normal image and reconstructs it
            autoencoder = Model(inputs=encoder_input, outputs=decoder)

            # Define an encoder that receives a normal image and outputs a representation of it in a different form
            encoder = Model(inputs=encoder_input, outputs=encoder)

            # Define an optimizer
            sigmoid = SGD(lr=self.lr, decay=self.decay, momentum=self.momentum)

            autoencoder.compile(optimizer=sigmoid, loss='binary_crossentropy')

            encoder.compile(optimizer=sigmoid, loss="binary_crossentropy")

            # Train the autoencoder and the encoder (they use the same layer reference)
            autoencoder.fit(data_represent_i, data_represent_i,
                            epochs=pre_epochs,
                            batch_size=self.batch_size,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=[EarlyStopping(monitor='val_loss',
                                                     min_delta=0, patience=0,
                                                     verbose=0, mode='auto')])

            # Get the output of the encoder to use it as input to the next autoencoder
            data_represent_i = encoder.predict(data_represent_i)

            pre_trained_weights[i] = encoder.get_weights()

        return pre_trained_weights

    def train(self, x_train, y_train, epochs, init_weights=None):
        """
        :param init_weights: Initial values for the weights of the model
        :param x_train: Training data
        :param y_train: Training targets
        :param epochs: Number of epochs to train
        :return: train history
        """
        # Initialize a new model
        self.model = Sequential()

        if init_weights is not None:
            # Initialize the weights of the layers to the given pre-trained weights
            for i in range(len(init_weights)):
                input_dim = init_weights[i][0].shape[0]
                nodes_of_layer = init_weights[i][0].shape[1]

                print("Setting layer " + str(i) + " with " + str(nodes_of_layer) +
                      " nodes" + " and input " + str(input_dim))

                self.model.add(Dense(nodes_of_layer, activation=self.h_act_function, input_dim=input_dim))

                self.model.layers[i].set_weights(init_weights[i])
        else:
            self.model.add(Dense(512, activation=self.h_act_function, input_dim=x_train.shape[1]))

            self.model.add(Dense(256, activation=self.h_act_function))

            self.model.add(Dense(128, activation=self.h_act_function))

        # Add an output layer at the end for classification
        self.model.add(Dense(10, activation=self.out_act_function))

        sigmoid = SGD(lr=self.lr, decay=self.decay, momentum=self.momentum)

        self.model.compile(optimizer=sigmoid, loss='MSE', metrics=['acc'])

        print(self.model.summary())

        # Train the DNN
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 batch_size=self.batch_size,
                                 validation_split=0.1,
                                 shuffle=True,
                                 verbose=1,
                                 callbacks=[EarlyStopping(monitor='val_loss',
                                                          min_delta=0, patience=0,
                                                          verbose=0, mode='auto')])
        return history

    def test(self, x_test, y_test, binary=True, verbose=1):
        """
        :param x_test: Testing data
        :param y_test: Testing labels
        :param binary: return output in a binary format
        :param verbose: Show testing info
        """
        predicted = self.model.predict(x_test, batch_size=self.batch_size, verbose=verbose)

        score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=2)

        if binary:
            predicted = np.where(predicted < 0.5, 0, 1)

        return predicted, score
