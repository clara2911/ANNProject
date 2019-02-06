#!/usr/bin/env python
"""
Main file for assignment 3
RBF Networks

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

class RBF_Net:


    class RBF_node(object):
        """
        A class that represents a RBF node
        """
        def __init__(self):
            """
            Initialize random center and std
            """
            # TODO: MAKE mu evenly separated within the X axis
            self.mu = np.random.normal(0, 1.)
            self.sigma = np.random.normal(0, 1.)


    def __init__(self, net_size, train_X):
        """
        Initialize a RBF Network
        """

        N = train_X.shape[0]  # Number of data points

        if (N > net_size):  # Number of data points should always be lower or equal to number of nodes
            print("System overdetermined. Cannot solve it using least squares.")
            exit()

        # Initialize RBF layer by adding randomly initialized nodes
        self.RBF_Layer = []
        for i in range(net_size):
            self.RBF_Layer.append(self.RBF_node())

        # Set which algorithm to use
        self.learning_method = {
            'least_squares' : self.lstsq(),
            'delta_rule' : self.delta_rule()
        }


    def train(self, train_X, train_Y, method):
        """
        Forward pass
        Backward pass
        N: number of samples
        n: number of nodes
        """
        # Convert data to column vectors
        train_X = train_X.reshape(-1, 1)
        train_Y = train_Y.reshape(-1, 1)

        algorithm = self.learning_method[method]
        return algorithm.update(self.RBF_Layer, train_X, train_Y)


    def transfer_function(x, mu_i, sigma_i):
        """
        Calculate the output of a Gaussian RBF node i
        """
        numerator = - ( x - mu_i) ** 2
        denominator = 2 * (sigma_i ** 2)
        exp_term = numerator / denominator
        return np.exp(exp_term)


    def calculate_out(phi, w):
        """
        Calculate an approximation of the output function given the output
        of the RBF nodes and the weights of the hidden layer
        """
        f_out = np.dot(phi, w)
        return f_out


    class lstsq(object):
        """
        A class containing all the necessary functions for the Least Squares Solution method
        """
        def update(self, RBF_Layer, train_X, train_Y):
            """
            Update the weights using batch learning and the least square solution
            """

            phi = self.calculate_phi(RBF_Layer, train_X)

            w = self.calculate_weights(phi, train_Y)

            y_pred = RBF_Net.calculate_out(phi, w)

            return y_pred


        def calculate_phi(self, RBF_Layer, train_X):
            """
            Calculate the output of the rbf nodes for every training sample
            """
            # Initialize the output of the RBF nodes
            phi = np.empty(shape=(train_X.shape[0], len(RBF_Layer)))

            # TODO: this can be optimized to not use 2 for loops, but pass it as a matrix
            for i, sample in enumerate(train_X):
                for j, node in enumerate(RBF_Layer):
                    phi[i][j] = RBF_Net.transfer_function(sample, node.mu, node.sigma)

            return phi


        def calculate_weights(self, phi, f):
            """
            Obtain w which minimizes the system phi.T * f_pred = phi.T * f
            """
            pseudo_inv_phi = np.dot(phi.T, phi)
            pseudo_inv_f = np.dot(phi.T, f)
            # NOTE: This is using a ready made function (Working correctly)
            w = np.linalg.lstsq(pseudo_inv_phi, pseudo_inv_f, rcond=None)[0]
            # NOTE: This is manually solving the equation (Currently(!) not working correctly)
            # pseudo_inv_phi = np.linalg.inv(pseudo_inv_phi)
            # w = np.dot(pseudo_inv_phi.T, pseudo_inv_f)

            return w


    class delta_rule(object):
        """
        A class containing all the necessary functions for the delta rule method
        """
        def update(self, RBF_Layer, train_X, train_Y):
            """
            Update the weights using sequential (online) learning and delta rule
            """
            return
