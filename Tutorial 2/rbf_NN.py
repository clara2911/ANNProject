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
        mu = 0.
        sigma = np.random.normal(0, 0.1)


    def __init__(self, net_size):
        """
        Initialize a RBF Network
        """

        # Initialize RBF layer by adding randomly initialized nodes
        self.RBF_Layer = []
        self.net_size = net_size
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
        """
        # Convert data to column vectors
        train_X, train_Y = train_X.reshape(-1, 1), train_Y.reshape(-1, 1)

        # Initialize the output of the RBF nodes
        rbf_out = np.empty(shape=(self.net_size, train_X.shape[0]))

        # Forward pass
        for i, node in enumerate(self.RBF_Layer):
            rbf_out[i] = self.transfer_function(train_X[0], node.mu, node.sigma)

        # Backward pass
        algorithm = self.learning_method[method]
        algorithm.update(rbf_out, train_Y)


    def transfer_function(self, x, mu_i, sigma_i):
        """
        Calculate the output of a Gaussian RBF node i
        """
        numerator = - ( x - mu_i) ** 2
        denominator = 2 * (sigma_i ** 2)
        exp_term = numerator / denominator
        return np.exp(exp_term)


    def calculate_out(self, rbf_out):
        """
        Calculate an approximation of the output function given the output
        of the RBF nodes and the weights of the hidden layer
        """
        f_out = np.dot(self.weights, rbf_out)


    class lstsq(object):
        """
        A class containing all the necessary functions for the Least Squares Solution method
        """
        def update(self, train_X, train_Y):
            """
            Update the weights using batch learning and the least square solution
            """
            N = train_X.shape[0]  # Number of data points
            n = len(RBF_Layer)  # Number of nodes
            if (N > n):  # Number of data points should always be lower or equal to number of nodes
                return 0

            y_pred = np.linalg.lstsq(r, train_Y, rcond=None)
            return y_pred


        def least_squares(self, f_pred, f):
            """
            Calculate the least squares error of the predicted and the true function
            """
            return


        def calculate_weights(self, rbf_out, f_pred, f):
            """
            Obtain w which minimizes the system rbf_out.T * f_pred = rbf_out.T * f
            """
            a = 1
            b = 1
            return np.lstsq(a, b)


    class delta_rule(object):
        """
        A class containing all the necessary functions for the delta rule method
        """
        def update(self, train_X, train_Y):
            """
            Update the weights using sequential (online) learning and delta rule
            """
            return
