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

def addNoise(X, std):
    '''
    :param X: a numpy array
    :param std: standard deviation
    :return: X with noise
    '''
    noise = np.random.normal(0, std, X.shape)
    noiseX = X + noise
    return noiseX

# def noise_layers(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, plot=True):
#
#     batch_size = 100
#     learning_rate = 0.001
#     h1 = 7
#     w_rel = 0.001
#     h2list = [3,4,5,6,7,8]
#     stds = [0.03,0.09,0.18]
#     epochs = 500
#     repeats = 10
#     colors=['red', 'blue', 'green']
#
#     for i in range(len(stds)):
#         std = stds[i]
#         Xtrain = addNoise(Xtrain, std)
#         color = colors[i]
#         val_errors = []
#         for h2 in h2list:
#             error_mean_val = np.zeros(len(range(repeats)))
#             for i in range(repeats):
#                 model = Sequential()
#                 model.add(Dense(h1, activation='relu', input_shape=(5,),
#                                 kernel_regularizer=regularizers.l2(w_rel), use_bias=False))
#                 model.add(Dense(h2, activation='relu',
#                                 kernel_regularizer=regularizers.l2(w_rel), use_bias=False))
#                 model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(w_rel), use_bias=False))
#
#                 # settings
#                 sgd = optimizers.Adam(lr=learning_rate)
#                 model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
#
#                 # training
#                 history = History()
#                 callbacks = [history, EarlyStopping(monitor='val_loss', patience=50),
#                              ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='auto')]
#                 model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xval, Yval), verbose=0,
#                           callbacks=callbacks)
#                 error_mean_val[i] = history.history['val_loss'][-1]
#
#
#             val_errors.append(error_mean_val)
#             print("h2: ", h2)
#             print("noise: ", std)
#             print("error validation")
#             print(error_mean_val)
#         val_error_means = [np.mean(error) for error in val_errors]
#         val_error_stds = [np.std(error) for error in val_errors]
#         print("PLOTTING")
#         print("X: ", h2list)
#         print("Y: ", val_error_means)
#         print("errors: ", val_error_stds)
#         plt.errorbar(h2list, val_error_means, yerr=val_error_stds, linestyle='--', ecolor='b', fmt='o', label='Noise std = ' + str(std), color=color)
#         plt.xlabel('Size hidden layer 2', fontsize=18)
#         plt.ylabel('Validation error', fontsize=18)
#         plt.legend()
#     plt.show()

def two_vs_three(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, plot=True):

    batch_size = 10
    learning_rate = 0.001
    h1 = 7
    w_rel = 0.001
    h2 = 4
    stds = [0, 0.03, 0.09, 0.18]
    epochs = 100
    repeats = 3
    colors = ['red', 'blue']
    noLayersList = [2,3]

    for k in range(len(noLayersList)):
        noLayers = noLayersList[k]
        val_errors = []
        for i in range(len(stds)):
            std = stds[i]
            Xtrain = addNoise(Xtrain, std)
            error_mean_val = np.zeros(len(range(repeats)))
            for j in range(repeats):
                model = Sequential()
                model.add(Dense(h1, activation='relu', input_shape=(5,),
                                kernel_regularizer=regularizers.l2(w_rel), use_bias=False))
                if noLayers == 3:
                    model.add(Dense(h2, activation='relu',
                                kernel_regularizer=regularizers.l2(w_rel), use_bias=False))
                model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(w_rel), use_bias=False))

                # settings
                sgd = optimizers.Adam(lr=learning_rate)
                model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

                # training
                history = History()
                callbacks = [history, EarlyStopping(monitor='val_loss', patience=50),
                             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True,
                                             mode='auto')]
                model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xval, Yval),
                          verbose=0,
                          callbacks=callbacks)
                error_mean_val[j] = history.history['val_loss'][-1]
            print("appending to val_errors: ")
            print(error_mean_val)
            val_errors.append(error_mean_val)
            # print("number of layers: ", noLayers)
            # print("noise: ", stds)
            # print("error validation")
            # print(error_mean_val)
        print("all validation errors")
        print(val_errors)
        val_error_means = [np.mean(error) for error in val_errors]
        val_error_stds = [np.std(error) for error in val_errors]
        print("PLOTTING")
        print("X: ", stds)
        print("Y: ", val_error_means)
        print("errors: ", val_error_stds)
        plt.errorbar(stds, val_error_means, yerr=val_error_stds, linestyle='--', ecolor='b', fmt='o',
                     label='Number of layers= ' + str(noLayers), color=colors[k])
        plt.xlabel('Std of added noise', fontsize=18)
        plt.ylabel('Validation error', fontsize=18)
        plt.legend()
    plt.show()



if __name__ == '__main__':

    batch_size = 10

    hidden_layers = 2
    hidden_neurons = 5
    w_rel = 0.01
    database = Database()
    inputs, outputs = database.build_data()
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = database.Split_data(inputs, outputs, train=700, val=300)

    two_vs_three(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, plot=False)
