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
        "epochs": 5000,
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
    recalled_set, energy = Hop.sequential_recall(recall_set)

    #print
    import matplotlib.pylab as plt

    rs0_real = P[0]
    rs0_org = P[9]
    rs0 = recalled_set[0, :].reshape(32, 32)
    rs1_real = P[1]
    rs1_org = P[10]
    rs1 = recalled_set[1, :].reshape(32, 32)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(rs0_real, origin="lower")
    ax[1].imshow(rs0_org, origin="lower")
    ax[2].imshow(rs0, origin="lower")
    plt.show()
    plt.plot(range(len(energy[0])), energy[0])
    plt.show()

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(rs1_real, origin="lower")
    ax[1].imshow(rs1_org, origin="lower")
    ax[2].imshow(rs1, origin="lower")
    plt.show()

    plt.plot(range(len(energy[1])), energy[1])
    plt.show()
    print("hola")


