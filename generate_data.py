import numpy as np
from matplotlib import pyplot as plt

class DataBase:

    def make_data(N, features, mA, mB, sigmaA, sigmaB, plot=False):
        n = int(N/2)
        #generate random
        classA = np.zeros((n, features))
        targetA = np.ones((n,1))
        classB = np.zeros((n, features))
        targetB = np.ones((n,1))*-1

        for feature in range(features):
            classA[:,feature] = np.random.normal(mA[feature], sigmaA, n)
            classB[:,feature] = np.random.normal(mB[feature], sigmaB, n)

        if (plot):
            plot_data(classA, classB)

        X = np.vstack((classA, classB))
        Y = np.vstack((targetA, targetB))

        return X, Y

    def make_non_lin_data(N, features, mA, mB, sigmaA, sigmaB, plot=False):
        pass

    def plot_data(classA, classB):
        plt.scatter(classA[0,:], classA[1,:], color='cyan', alpha=0.7)
        plt.scatter(classB[0,:], classB[1,:], color='purple', alpha=0.7)
        plt.show()
