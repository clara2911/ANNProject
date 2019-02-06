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

    batch_size = 100
    hidden_neurons = 10
    output_neurons = 1


    def __init__(self):
        """
        Initialize NN settings
        """



    def solve(self, train_X, train_Y, test_X, test_Y):
        """
        Initialize NN, train and test
        """
        model = setup_model(train_X)

        trained_model = train(model, train_X, train_Y)

        y_pred = test(trained_model, test_X, test_Y)


    def setup_model(self, train_X):
        """
        Initialize model with parameters
        """

        input_dims = train_X.shape[0]
        model = Sequential()
        model.add(Dense(hidden_neurons, activation='relu', input_shape=(input_dims,)), kernel_regularizer=regularizers.l2(0.01))
        model.add(Dense(output_neurons, activation='linear'), kernel_regularizer=regularizers.l2(0.01))

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
        model.fit(train_X, train_Y, epochs=5000, batch_size=batch_size, verbose=1, callbacks=callbacks)
        return model


    def test(self, model, test_X, test_Y):
        """
        Test NN on unseen data
        """
        loss_and_metrics = model.evaluate(test_X, test_Y, batch_size=batch_size)
        # Predict
        test_Y = model.predict(test_X, batch_size=batch_size, verbose=0, steps=None)
        return test_Y
