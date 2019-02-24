import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras import optimizers
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.initializers import RandomNormal
import data
class Autoencoder:

    def __init__(self, **kwargs):
        """
        Initialize algorithm with data and parameters
        """
        var_defaults = {
            "bias_init" : 'zeros',
            "weight_init" : [0.0, 0.1],
            "seed" : None,
            "num_hid_nodes" : 32,
            "activations": 'sigmoid',
            "lr" : 0.01,
            "decay": 0,
            "momentum" : 0,
            "nesterov" : False,
            "loss" : 'mean_squared_error',
            "epochs" : 5,
            "batch_size" : 256
        }
        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

    def train(self, x_train, x_test):
        dim = x_train.shape[1]
        input_img = Input(shape=(dim,))
        rand_norm = RandomNormal(mean=self.weight_init[0], stddev=self.weight_init[1], seed=self.seed)
        encoded = Dense(self.num_hid_nodes, activation=self.activations,
                        kernel_initializer=rand_norm,
                        bias_initializer=self.bias_init)(input_img)
        #last layer should always be sigmoid so that out is in [0,1]
        decoded = Dense(dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_img, decoded)
        sgd = optimizers.SGD(lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=self.nesterov)
        autoencoder.compile(loss=self.loss, optimizer=sgd)
        autoencoder.fit(x_train, x_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test))
