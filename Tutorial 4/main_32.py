#!/usr/bin/env python
"""
Main file for Tutorial 4 Part 3.2

Authors: Kostis SZ, Romina Arriaza and Clara Tump
"""

import datetime
import json
import data
import plot
from dnn import DNN

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


os_slash = '/'  # Unix: '/' Windows: '\'

now = datetime.datetime.now()
MODEL_FOLDER = os.getcwd() + os_slash + "m_" + str(now.day) + "." + str(now.month) + "_" + str(now.hour) + ":" + str(now.minute)
os.makedirs(MODEL_FOLDER)

params = {
    "lr": 0.15,
    "decay": 0, #1e-4,
    "momentum": 0, # 0.7,
    "h_act_function": 'relu',
    "out_act_function": 'sigmoid'
}

limit_memory = True
memory_use = 0.35


def main():
    x_train, y_train, x_test, y_test = data.mnist(one_hot=True)

    # Define Deep Neural Network structure (input_dim, num_of_nodes)
    layers = [
        [x_train.shape[1], 256],
        [256, 128],
        [128, 64]
    ]

    # Initialize a deep neural network

    dnn = DNN(MODEL_FOLDER, os_slash, layers, params)

    pre_epochs = 100
    train_epochs = 100

    # Create auto-encoders and train them one by one by stacking them in the DNN
    pre_trained_weights = dnn.pre_train(x_train, pre_epochs)

    # Then use the pre-trained weights of these layers as initial weight values for the MLP
    history = dnn.train(x_train, y_train, train_epochs, init_weights=pre_trained_weights)

    plot.plot_loss(history, loss_type='MSE')

    predicted, score = dnn.test(x_test, y_test)

    print("Test accuracy: ", score[1])

    dnn.model.save_weights(MODEL_FOLDER + os_slash + "final_weights.h5")
    dnn.model.save(MODEL_FOLDER + os_slash + "model.h5")
    save_results(score[1])


def save_results(accuracy):
    """
    Save to a unique folder the parameters and results of the model
    """
    with open(MODEL_FOLDER + os_slash + 'model_params.json', 'w') as f:
        json.dump(params, f)

    with open(MODEL_FOLDER + os_slash + 'results.out', 'w') as f:
        f.write("Test accuracy: " + str(accuracy))


def config_mem():
    """
    Tensorflow by default allocates all memory available. Limit its memory use by percentage
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory_use
    set_session(tf.Session(config=config))
    print("USING " + str(memory_use) + "% OF MEMORY")


if __name__ == "__main__":
    if limit_memory:
        config_mem()
    main()
