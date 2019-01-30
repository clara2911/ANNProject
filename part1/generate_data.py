"""
Generate data for different purposes

Authors: Clara Tump, Kostis SZ... and Romina Azz...
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DataBase:

    def __init__(self):
        self.function = 'Hello, use me for plotting!'


    def plot_data(self, classA, classB):
        """
        This function just plot the data with scatter, raw data, no-boundaries
        """
        plt.scatter(classA[:,0], classA[:,1], color='cyan', alpha=0.7, s=7)
        plt.scatter(classB[:,0], classB[:,1], color='purple', alpha=0.7, s=7)
        plt.axis('tight')
        plt.show()


    def make_data(self, n, features, mA, mB, sigmaA, sigmaB, plot=False, add_bias=False):
        """
        This functions creates data for 2 classes with a
        gaussian distribution given the parameters in args.
        """
        #generate random
        classA = np.zeros((n, features))
        targetA = np.ones((n,1))
        classB = np.zeros((n, features))
        targetB = np.ones((n,1))*-1

        for feature in range(features):
            classA[:,feature] = np.random.normal(mA[feature], sigmaA, n)
            classB[:,feature] = np.random.normal(mB[feature], sigmaB, n)

        if plot:
            self.plot_data(classA, classB)

        X = np.vstack((classA, classB))
        Y = np.vstack((targetA, targetB))

        X, Y = self.shuffle(X, Y)
        if (add_bias):
            X = self.add_bias_to_input(X)
        return X, Y


    def shuffle(self, X, Y):
        """
        Shuffle data
        """
        index_shuffle = np.random.permutation(X.shape[0])  # shuffle indices
        X = X[index_shuffle]
        Y = Y[index_shuffle]
        return X, Y


    def add_bias_to_input(self, X):
        """
        Add bias as input vector (feature) in the data
        """
        bias = - np.ones((X.shape[0], 1))  # bias is: -1 !
        X = np.hstack((X, bias))  # changed so bias is after (before it was beginning)
        return X


    def non_linear_data(self, sampleA = 1.0, sampleB = 1.0, subsamples = False, add_bias=False):
        """
        :param sampleA: percentage of data from class A that is going to be used. 100% = 100 samples.
        :param sampleB: percentage of data from class B that is going to be used. 100% = 100 samples.
        :param subsamples: boolean variable that indicates to the function to make a
        special sample to cover part 4 of question 3.1.3
        :return: samples from class A and B

        This program generates data according to the parameters indicated by 3.1.3.
        Then It samples randomly the amount of data according to sampleA and sample B (fractions).
        """

        ndata = 100
        mA = [1.0, 0.3]
        sigmaA = 0.2
        mB = [0.0, -0.1]
        sigmaB = 0.3

        classA = np.empty((ndata, 2))
        classB = np.empty((ndata, 2))
        classA[:, 1] = np.random.randn(ndata) * sigmaA + mA[1]
        classA[:, 0] = np.hstack((np.random.randn(int(ndata/2)) * sigmaA - mA[0], np.random.randn(int(ndata/2)) * sigmaA + mA[0]))
        classB[:, 0] = np.random.randn(ndata) * sigmaB + mB[0]
        classB[:, 1] = np.random.randn(ndata) * sigmaB + mB[1]

        targetA = np.ones((ndata,1))
        targetB = np.ones((ndata,1))*-1

        #sample
        if subsamples: #special case
            pos_ind = np.where(classA[:, 0] > 0)[0]
            neg_ind = np.where(classA[:, 0] < 0)[0]
            indA_pos = np.random.choice(pos_ind, int(0.2 * len(pos_ind)), replace=False)
            indA_neg = np.random.choice(neg_ind, int(0.8 * len(neg_ind)), replace=False)
            indA = np.hstack((indA_pos, indA_neg))
            indB = range(len(targetB))
        else: #general case
            indA = np.random.choice(range(len(targetA)), int(sampleA * len(targetA)), replace=False)
            indB = np.random.choice(range(len(targetB)), int(sampleB * len(targetB)), replace=False)

        classA = classA[indA, :]
        classB = classB[indB, :]
        targetA = targetA[indA]
        targetB = targetB[indB]

        X = np.vstack((classA, classB))
        Y = np.vstack((targetA, targetB))

        X, Y = self.shuffle(X, Y)
        if (add_bias):
            X = self.add_bias_to_input(X)

        return X, Y

    def make_3D_data(self, bias = True, plot_data=False):
        x = np.arange(-5, 5.5, 0.5)
        y = np.arange(-5, 5.5, 0.5)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(- X**2 * 0.1) * np.exp(- Y**2 * 0.1) - 0.5

        #plot the objective function
        if (plot_data):
            ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            ax.set_title('surface')
            plt.show()

        ndata = len(x)*len(x)
        targets = Z.reshape((ndata, 1))
        patterns = np.hstack((X.reshape((ndata,1)), Y.reshape((ndata,1))))
        if bias:
            patterns = np.hstack((patterns, np.ones((len(targets), 1))*-1))

        return patterns, targets

    # make one-hot encoding matrix where pos=1 and neg=-1 of length N
    def one_hot(self, N, pos, neg):
        X = neg*np.ones([N,N])
        X[range(N), range(N)] = pos
        return X
