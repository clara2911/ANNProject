import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras import optimizers
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.initializers import RandomNormal
import data
from auto_encoder import Autoencoder

np.random.seed(10)

def main():
    x_train, y_train, x_test, y_test = data.mnist()

    params = {
            "epochs" : 5
        }
    auto_enc1 = Autoencoder(**params)
    auto_enc1.train(x_train, x_test)



if __name__ == "__main__":
    main()
