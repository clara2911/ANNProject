import numpy as np
from matplotlib import pyplot as plt

class DataBase:

    def __init__(self):
        self.function = 'Hello, use me for plotting!'


    def make_data(self, n, features, mA, mB, sigmaA, sigmaB, plot=False, add_bias=True):
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

    def plot_data(self, classA, classB):
        plt.scatter(classA[0,:], classA[1,:], color='cyan', alpha=0.7)
        plt.scatter(classB[0,:], classB[1,:], color='purple', alpha=0.7)
        plt.show()
