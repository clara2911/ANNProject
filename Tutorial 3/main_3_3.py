#!/usr/bin/env python
"""
hopfield_net.py
Implements a Hopfield network

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

from hopfield_net import HopfieldNet
import data
import numpy as np

if __name__ == '__main__':

    P = data.load_file()
    params = {
        "epochs": 100,
        "neurons": 1024,
        "learn_method": 'classic'
    }

    p0 = P[0].ravel()
    p1 = P[1].ravel()
    p2 = P[2].ravel()
    train_set = np.vstack((p0, p1))
    train_set = np.vstack((train_set, p2))

    Hop = HopfieldNet(train_set, **params)
    Hop.batch_train()

    p10 = P[9].ravel()
    p11 = P[10].ravel()
    recall_set = np.vstack((p10, p11))
    recalled_set = Hop.recall(recall_set)
    print("hola")


