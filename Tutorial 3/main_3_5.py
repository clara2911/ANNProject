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
    recalled_set[0], energy = Hop.sequential_recall_shuffle(recall_set, epochs=2)

    for i in add_p.keys():
        print(i)
        train_set = np.vstack((train_set, add_p[i]))
        Hop = HopfieldNet(train_set)
        Hop.batch_train()
        recall_set = np.vstack((recall_set, add_p[i]))
        recalled_set[i+1], energy = Hop.sequential_recall_shuffle(recall_set, epochs=2)

    error_pattern = {}
    error_pattern[0] = []
    error_pattern[1] = []
    error_pattern[2] = []
    for i in recalled_set.keys():
        error_pattern[0] += [abs(np.mean(recalled_set[i][0, :] - p0))]
        error_pattern[1] += [abs(np.mean(recalled_set[i][1, :] - p1))]
        error_pattern[2] += [abs(np.mean(recalled_set[i][2, :] - p2))]

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(recalled_set[i][0, :].reshape(32, 32), origin="lower")
        ax[1].imshow(recalled_set[i][1, :].reshape(32, 32), origin="lower")
        ax[2].imshow(recalled_set[i][2, :].reshape(32, 32), origin="lower")
        plt.show()

    plt.plot(range(len(error_pattern[0])), np.array(error_pattern[0]))
    plt.plot(range(len(error_pattern[1])), np.array(error_pattern[1]))
    plt.plot(range(len(error_pattern[2])), np.array(error_pattern[2]))
    plt.xlabel('Number of patterns added', fontsize=16)
    plt.ylabel('Error %', fontsize=16)
    plt.show()

def second():
    params = {
        "epochs": 4000,
        "neurons": 100, #1024,
        "learn_method": 'classic'
    }

    P = {}
    for i in range(300):
        P[i] = np.random.randint(-1, 1, params['neurons']).reshape(1,-1)
        ind = np.where(P[i]==0)[1]
        P[i][0][ind] = 1

    # train
    train_set = P[0].ravel()
    for i in np.arange(1, 11):
        train_set = np.vstack((train_set, P[i].ravel()))
    Hop = HopfieldNet(train_set)
    Hop.batch_train()

    #recover
    n = int(300*0.1)
    recall_set = np.copy(train_set)
    for i in range(recall_set.shape[0]):
        ind = np.random.randint(0, len(recall_set[i,:]), n)
        recall_set[i, ind] = recall_set[i, ind] * (-1)

    recalled_set = {}
    recovered = []
    print(0)
    recalled_set[0], energy = Hop.sequential_recall_shuffle(recall_set, epochs=50)
    error = np.sum(recalled_set[0] - train_set, axis = 1)
    num_rec = len(np.where(error == 0.0)[0])
    recovered += [num_rec]

    count = 1
    for i in np.arange(11, len(P)):
        print(i)
        train_set = np.vstack((train_set, P[i].ravel()))
        Hop = HopfieldNet(train_set)
        Hop.batch_train()

        recall_set = np.copy(train_set)
        for k in range(recall_set.shape[0]):
            ind = np.random.randint(0, len(recall_set[k, :]), n)
            recall_set[k, ind] = recall_set[k, ind]*(-1)

        #recall_set = train_set
        if i%10==0:
            recalled_set[count], energy = Hop.sequential_recall_shuffle(recall_set, epochs=50)
            #error = np.sqrt(np.sum((recalled_set[0] - train_set) ** 2, axis=1))

            error = np.sum(recalled_set[count] - train_set, axis=1)
            num_rec = len(np.where(error == 0.0)[0])
            recovered += [num_rec]

            count += 1

    plt.plot(range(len(recovered)), np.array(recovered))
    plt.xticks(range(len(recovered)), np.arange(0, len(recovered)) + 11)
    plt.show()

    # errors = {}
    # for i in np.arange(0, 11):
    #     errors[i] = []
    # for j in range(len(recalled_set)):
    #     rs = recalled_set[j]
    #     for i in range(rs.shape[0]):
    #         errors[i] += [np.sqrt(np.sum((rs[i,:]-train_set[i,:])**2))]
    #
    # for i in errors.keys():
    #     plt.plot(range(len(errors[i])), np.array(errors[i]))
    # plt.show()



if __name__ == '__main__':
    #first()
    second() # [11, 21, 41, 51, 54, 32, 20] 1024


