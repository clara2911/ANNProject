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

def first():
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

    Hop = HopfieldNet(train_set)
    Hop.batch_train()
    recall_set = np.vstack((p0, p1))
    recall_set = np.vstack((recall_set, p2))

    # ====== add one more =========================================
    p3 = P[3].ravel()
    p4 = P[4].ravel()
    p5 = P[5].ravel()
    p6 = P[6].ravel()
    add_p = {0: p3, 1:p4, 2:p5, 3:p6}

    recalled_set = {}

    Hop = HopfieldNet(train_set)
    Hop.batch_train()
    recalled_set[0] = Hop.recall(recall_set)

    for i in add_p.keys():
        print(i)
        train_set = np.vstack((train_set, add_p[i]))
        Hop = HopfieldNet(train_set)
        Hop.batch_train()
        recall_set = np.vstack((recall_set, add_p[i]))
        recalled_set[i+1] = Hop.recall(recall_set)

    for i in recalled_set.keys():
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(recalled_set[i][0, :].reshape(32, 32), origin="lower")
        ax[1].imshow(recalled_set[i][1, :].reshape(32, 32), origin="lower")
        ax[2].imshow(recalled_set[i][2, :].reshape(32, 32), origin="lower")
        plt.show()

def second():
    params = {
        "epochs": 4000,
        "neurons": 1024,
        "learn_method": 'classic'
    }

    P = {}
    for i in range(150):
        P[i] = np.random.randn(1, params['neurons'])*2 - 1

    # train
    train_set = P[0].ravel()
    Hop = HopfieldNet(train_set)
    Hop.batch_train()

    #recover
    recall_set = train_set
    recalled_set = {}
    recalled_set[0] = Hop.recall(recall_set)

    errors = {}
    count = 0
    for i in np.arange(1, len(P)):
        train_set = np.vstack((train_set, P[i].ravel()))
        Hop = HopfieldNet(train_set)
        Hop.batch_train()

        recall_set = train_set
        recalled_set[i] = Hop.recall(recall_set)

        if i%20 == 0:
            error = []
            for j in range(len(recalled_set)):
                error += [np.sqrt(np.mean((recalled_set[j]-train_set[j,:])**2))]
            error = np.array(error)
            errors[count] = error
            count += 1

    for i in errors.keys():
        plt.plot(range(len(errors[i])), errors[i])
    plt.show()



if __name__ == '__main__':
    second()
