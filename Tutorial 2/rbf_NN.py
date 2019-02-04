#!/usr/bin/env python
"""
Main file for assignment 3
RBF Networks

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

class RBF_Net:

    RBF_Layer = []

    method = {
        'least_squares' : self.lstsq(),
        'delta_rule' : self.delta_rule()
    }

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

        for i in range(net_size):
            RBF_Layer.append(RBF_node())


    def train(self, train_X, train_Y, method):
        """
        Forward pass
        Backward pass
        """
        # Initialize the output of the RBF nodes
        rbf_out = np.zeros((len(RBF_Layer)))
        # Forward pass
        for i, node in enumarate(RBF_Layer):
            rbf_out[i] = self.transfer_function(train_X, node.mu, node.sigma)

        # Backward pass

    def calculate_out(self, rbf_out):
        """
        Calculate an approximation of the output function given the weights of
        the hidden layer and the output of the RBF nodes
        """
        f_out = np.dot(self.weights, rbf_out)


    def transfer_function(self, x, mu_i, sigma_i):
        """
        Calculate the output of a Gaussian RBF node i
        """
        numerator = - ( x - mu_i) ** 2
        denominator = 2 * (sigma_i ** 2)
        exp_term = numerator / denominator
        return np.exp(exp_term)


    class lstsq(object):
        """
        A class containing all the necessary functions for the Least Squares Solution method
        """
        def train(self, train_X, train_Y):
            """
            Update the weights using batch learning and the least square solution
            """

            N = train_X.shape[0]  # Number of data points
            n = len(RBF_Layer)  # Number of nodes
            if (N > n):  # Number of data points should always be lower or equal to number of nodes
                return 0

            for x_i in train_X:

            return

        def least_squares(self, f_pred, f):
            """
            Calculate the least squares error of the predicted and the true function
            """
            return


        def calculate_weights(self, rbf_out, f_pred, f):
            """
            Obtain w which minimizes the system rbf_out.T * f_pred = rbf_out.T * f
            """
            a =
            b =
            return np.lstsq(a, b)



    class delta_rule(object):
        """
        A class containing all the necessary functions for the delta rule method
        """
        def train(self, train_X, train_Y):
            """
            Update the weights using sequential (online) learning and delta rule
            """
            return
