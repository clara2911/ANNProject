import keras
import numpy as np
from keras.models import Sequential
# General type layers
from keras.layers import Input, Dense, Dropout, Activation, Flatten
# CNN type layers
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.utils import np_utils
from matplotlib import pyplot as plt
mnist = keras.datasets.mnist
np.random.seed(10)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = np.where(x_train / 255 > 0.5, 0,1), np.where(x_test / 255 > 0.5,0,1)

print(x_train.shape)
plt.imshow(x_train[5])
plt.show()

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
