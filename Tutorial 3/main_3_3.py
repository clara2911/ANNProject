#!/usr/bin/env python
"""
hopfield_net.py
Implements a Hopfield network

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

from hopfield_net import HopfieldNet
import data
import numpy as np
import matplotlib.pylab as plt

def first_three():
    P = data.load_file()
    params = {
        "epochs": 4000,
        "neurons": 1024,
        "learn_method": 'classic'
    }

    # ====== train ========================================
    p0 = P[0].ravel()
    p1 = P[1].ravel()
    p2 = P[2].ravel()
    train_set = np.vstack((p0, p1))
    train_set = np.vstack((train_set, p2))

    Hop = HopfieldNet(train_set, **params)
    Hop.batch_train()

    # get energy per pattern
    energy_p0 = Hop.energy(p0, threshold=0)
    energy_p1 = Hop.energy(p1, threshold=0)
    energy_p2 = Hop.energy(p2, threshold=0)
    print('The enegy for p0 is: {}'.format(energy_p0))
    print('The enegy for p1 is: {}'.format(energy_p1))
    print('The enegy for p2 is: {}'.format(energy_p2))
    print('\n')

    # ====== test =========================================
    p10 = P[9].ravel()
    p11 = P[10].ravel()

    # get energy per pattern
    energy_p10 = Hop.energy(p10, threshold=0)
    energy_p11 = Hop.energy(p11, threshold=0)
    print('The enegy for distorted p10 is: {}'.format(energy_p10))
    print('The enegy for distorted p11 is: {}'.format(energy_p11))
    print('\n')

    recall_set = np.vstack((p10, p11))
    recalled_set, energy = Hop.sequential_recall(recall_set)

    rs0_real = P[0]
    rs0_org = P[9]
    rs0 = recalled_set[0, :].reshape(32, 32)
    rs1_real = P[2]
    rs1_org = P[10]
    rs1 = recalled_set[1, :].reshape(32, 32)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(rs0_real, origin="lower")
    ax[1].imshow(rs0_org, origin="lower")
    ax[2].imshow(rs0, origin="lower")
    plt.show()
    plt.plot(range(len(energy[0])), energy[0])
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Epoch', fontsize=16)
    plt.show()

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(rs1_real, origin="lower")
    ax[1].imshow(rs1_org, origin="lower")
    ax[2].imshow(rs1, origin="lower")
    plt.show()

    plt.plot(range(len(energy[1])), energy[1])
    plt.show()
    print("hola")

def last_two():
    P = data.load_file()
    params = {
        "epochs": 4000,
        "neurons": 1024,
        "learn_method": 'classic'
    }

    # generate weight matrix
    W = np.random.randn(params['neurons'], params['neurons'])
    #W = (W + W.T) / 2
    p = np.random.randint(0, 1, params['neurons']).reshape(1,-1)
    p = (p*2) - 1
    Hop = HopfieldNet(p, **params)
    Hop.W = W
    recalled_set, energy = Hop.sequential_recall(p)
    plt.imshow(recalled_set.reshape(32,32))
    plt.show()
    plt.plot(range(len(energy[0])), energy[0])
    plt.show()


if __name__ == '__main__':
    #first_three()
    last_two()
