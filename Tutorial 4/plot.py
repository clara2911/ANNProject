#!/usr/bin/env python
"""
File for all plotting functions

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def plot_loss(history, loss_type='MSE'):
    epochs = list(range(1, len(history.history['loss']) + 1))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, train_loss, c='b', label=str(loss_type) + ' Train set. Final: ' +str(round(train_loss[-1],3)))
    plt.plot(epochs, val_loss, c='g', label=str(loss_type) + ' Validation set. Final: ' + str(round(val_loss[-1],3)))
    plt.title(str(loss_type)+ " per epoch")
    plt.xlabel('Epoch')
    plt.ylabel(str(loss_type))
    plt.legend()
    plt.show()

def plot_images(true_ims, reconstr_ims):
    square_dim = int(math.sqrt(true_ims.shape[1]))
    true_ims = true_ims.reshape(true_ims.shape[0], square_dim, square_dim)
    reconstr_ims = reconstr_ims.reshape(reconstr_ims.shape[0], square_dim, square_dim)

    plt.figure()
    for i in range(1,len(true_ims)+1):
        plt.subplot(2,len(true_ims),i)
        plt.imshow(true_ims[i-1])
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                        right=False, left=False, labelleft=False)
    for j in range(1,len(true_ims)+1):
        plt.subplot(2,len(true_ims),len(true_ims)+j)
        plt.imshow(reconstr_ims[j-1])
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                        right=False, left=False, labelleft=False)
    plt.tight_layout()
    plt.suptitle("Reconstructed images")
    plt.show()