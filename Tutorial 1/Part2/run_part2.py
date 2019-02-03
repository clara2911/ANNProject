#!/usr/bin/env python
"""
Main file for learning with a single layer perceptron
It contains X tests:


Authors: Clara Tump, Kostis SZ and Romina Azz
"""

from Database import Database
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
import numpy as np
#np.random.seed(100)

def get_error_evolution(path, Xval, Yval):
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(5,)))
    # model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='relu'))
    return


if __name__ == '__main__':

    batch_size = 100

    #create database
    database = Database()
    inputs, outputs = database.build_data()
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = database.Split_data(inputs, outputs, train=700, val=300)

    #network setup
    hidden_layers = 2
    hidden_neurons = [5, 5]
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(5,)))#, kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Dense(5, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='linear'))#, kernel_regularizer=regularizers.l2(0.01)))

    #settings
    sgd = optimizers.Adam(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    #training
    history = History()
    callbacks = [history, EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='auto')]
    #EarlyStopping(monitor='val_loss', patience=10),
    model.fit(Xtrain, Ytrain, epochs=5000, batch_size=batch_size,
              validation_data=(Xval, Yval), verbose=1, callbacks=callbacks)
    loss_and_metrics = model.evaluate(Xtest, Ytest, batch_size=batch_size)

    #predict
    Ypred = model.predict(Xtest, batch_size=batch_size, verbose=0, steps=None)

    # #plot result best model
    # best_model = Sequential()
    # best_model.add(Dense(5, activation='relu', input_shape=(5,)))#, kernel_regularizer=regularizers.l2(0.01)))
    # #best_model.add(Dense(5, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    # best_model.add(Dense(1, activation='linear'))#, kernel_regularizer=regularizers.l2(0.01)))
    # best_model.load_weights('best_model.h5')
    # Ypred = best_model.predict(Xtest, batch_size=batch_size, verbose=0, steps=None)

    plt.plot(range(len(Ytest)), Ytest, 'b', range(len(Ytest)), Ypred, 'r')
    plt.legend(label=['real', 'estimation'], frameon=False, fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.show()

    #plot history
    plt.plot(history.epoch, history.history['loss'], history.epoch, history.history['val_loss'])
    plt.legend(['Training', 'Validation'], frameon=False, fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.show()

    print('hola')
