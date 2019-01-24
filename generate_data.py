import numpy as np
from matplotlib import pyplot as plt

class DataBase:

    def __init__(self):
        self.function = 'Hello, use me for plotting!'

    def plot_data(self, classA, classB):
        plt.scatter(classA[0,:], classA[1,:], color='cyan', alpha=0.7, s=7)
        plt.scatter(classB[0,:], classB[1,:], color='purple', alpha=0.7, s=7)
        plt.show()

    def make_data(self, n, features, mA, mB, sigmaA, sigmaB, plot=False):
        #generate random
        classA = np.zeros((n, features))
        targetA = np.ones((n,1))
        classB = np.zeros((n, features))
        targetB = np.ones((n,1))*-1

        for feature in range(features):
            classA[:,feature] = np.random.normal(mA[feature], sigmaA, n)
            classB[:,feature] = np.random.normal(mB[feature], sigmaB, n)

        if (plot):
            self.plot_data(classA, classB)

        X = np.vstack((classA, classB))
        Y = np.vstack((targetA, targetB))

        return X, Y

    def non_linear_data(self, sampleA = 1.0, sampleB = 1.0, subsamples = False):

        ndata = 100
        mA = [1.0, 0.3]
        sigmaA = 0.2
        mB = [0.0, -0.1]
        sigmaB = 0.3

        classA = np.empty((2, ndata))
        classB = np.empty((2, ndata))
        classA[0, :] = np.random.randn(ndata) * sigmaA - mA[0]
        classA[1, :] = np.random.randn(ndata) * sigmaA - mA[1]
        classB[0, :] = np.random.randn(ndata) * sigmaB - mB[0]
        classB[1, :] = np.random.randn(ndata) * sigmaB - mB[1]

        targetA = np.ones(ndata)
        targetB = np.ones(ndata)*-1

        #sample
        if subsamples:
            pos_ind = np.where(classA[1, :] > 0)[0]
            neg_ind = np.where(classA[1, :] < 0)[0]
            indA_pos = np.random.choice(pos_ind, int(0.2 * len(pos_ind)), replace=False)
            indA_neg = np.random.choice(neg_ind, int(0.8 * len(neg_ind)), replace=False)
            indA = np.hstack((indA_pos, indA_neg))
            indB = range(len(targetB))
        else:
            indA = np.random.choice(range(len(targetA)), int(sampleA * len(targetA)), replace=False)
            indB = np.random.choice(range(len(targetB)), int(sampleB * len(targetB)), replace=False)

        classA = classA[:, indA]
        classB = classB[:, indB]
        targetA = targetA[indA]
        targetB = targetB[indB]

        X = np.hstack((classA, classB))
        Y = np.hstack((targetA, targetB))

        return X.T, Y.reshape(-1,1)
