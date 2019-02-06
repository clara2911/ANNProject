#!/usr/bin/env python
"""
A 2 layer (1-hidden, 1-output) neural network

Authors: Clara Tump, Kostis SZ and Romina Azz
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History


class ANN:

    def __init__(self, epochs, batch_size, hidden_neurons, output_neurons):
        """
        Initialize NN settings
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons


    def solve(self, train_X, train_Y, test_X, test_Y):
        """
        Initialize NN, train and test
        """
        model = self.setup_model(train_X)

        trained_model = self.train(model, train_X, train_Y)

        y_pred = self.test(trained_model, test_X, test_Y)

        return y_pred


    def setup_model(self, train_X):
        """
        Initialize model with parameters
        """

        input_dims = 1
        model = Sequential()
        model.add(Dense(self.hidden_neurons, activation='relu', input_shape=(input_dims,)))
        model.add(Dense(self.output_neurons, activation='linear'))

        sgd = optimizers.Adam(lr=0.01)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

        return model


    def train(self, model, train_X, train_Y):
        """
        Train NN on training data
        """
        # Initialize Model Checkpint
        history = History()
        callbacks = [history, ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='auto')]

        # Train
        model.fit(train_X, train_Y, epochs=self.epochs, batch_size=self.batch_size, verbose=1, callbacks=callbacks)
        return model


    def test(self, model, test_X, test_Y):
        """
        Test NN on unseen data
        """
        loss_and_metrics = model.evaluate(test_X, test_Y, batch_size=self.batch_size)
        # Predict
        test_Y = model.predict(test_X, batch_size=self.batch_size, verbose=0, steps=None)
        return test_Y
