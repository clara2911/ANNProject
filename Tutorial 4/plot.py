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

def plot_parameter(param_name, lr_list, error_list, std_list):
    plt.errorbar(lr_list, error_list, yerr=std_list, color='blue', ecolor='red')
    plt.title(param_name)
    plt.xlabel(param_name)
    plt.ylabel('MSE')
    plt.show()

def plot_losses(layer_sizes, sigmoid_histories, relu_histories):
    end_range = len(sigmoid_histories[0]) + 1
    epochs = list(range(1, end_range ))
    for i in range(len(sigmoid_histories)):
        print("epochs: ", epochs)
        print("sigmoid_histories[i]: ", sigmoid_histories[i])
        plt.errorbar(epochs, sigmoid_histories[i], label='sigmoid // '+str(layer_sizes[i]) + ' nodes')
        plt.errorbar(epochs, relu_histories[i], label='relu // '+str(layer_sizes[i]) + ' nodes')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('RELU vs Sigmoid')
    plt.legend()
    plt.show()
    
def plot_weights_image(auto_encoder, nodes, num_w):
    weights = auto_encoder.get_weights()
    for i, layer in enumerate(auto_encoder.autoencoder._layers):
        if i != 0:
            all_weights = layer.get_weights()
            weights = np.zeros((nodes, num_w))
            for j, w in enumerate(all_weights[0]):
                for k, elem in enumerate(np.array(w)):
                    weights[j,k] = elem
            ind = np.random.randint(0, weights.shape[1], 25)

            fig, ax = plt.subplots(5,5)
            for k,j in enumerate(ind):
                im_w = weights[:, j].reshape(28,28)
                row = int(k/5)
                col = int(k%5)
                ax[row, col].imshow(im_w)
            fig.suptitle('Weights {} layer'.format(i), fontsize = 14)
            plt.show()
    print("hola")

def plot_weights_hist(auto_encoder):
    for i, layer in enumerate(auto_encoder.autoencoder._layers):
        if i != 0:
            all_weights = layer.get_weights()
            weights = []
            for w in all_weights[0]:
                weights = np.hstack((weights, w))

            n, bins, patches = plt.hist(np.array(weights), 20, density=False, facecolor='g', alpha=0.75)
            plt.xlabel('Bins', fontsize = 14)
            plt.ylabel('Count', fontsize = 14)
            plt.title('Histogram of weights {} layer'.format(i), fontsize = 14)
            plt.grid(True)
            plt.show()
