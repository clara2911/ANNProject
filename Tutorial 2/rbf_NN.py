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
        def __init__(self, mu, sigma):
            """
            Initialize random center and std
            """
            self.mu = mu
            self.sigma = 0.5


    def __init__(self, net_size, train_X, sigma=0.5):
        """
        Initialize a RBF Network
        """
        # Set which algorithm to use
        self.learning_method = {
            'least_squares' : self.lstsq,
            'delta_rule' : self.delta_rule
        }

        self.RBF_Layer = []
        # Initialize mu's evenly spaced in the x axis of the data and random sigma's
        mu_step = train_X[-1] / net_size
        for i in range(net_size):
            mu_i = i * mu_step
            self.RBF_Layer.append(self.RBF_node(mu_i, sigma))

        self.weights = None
        self.phi = None


    def train(self, train_X, train_Y, method, lr=None):
        """
        Forward pass
        Backward pass
        N: number of samples
        n: number of nodes
        """
        # Convert data to column vectors
        train_X = train_X.reshape(-1, 1)
        train_Y = train_Y.reshape(-1, 1)

        phi = self.calculate_phi(train_X)

        algorithm = self.learning_method[method]
        self.weights = algorithm(phi, train_Y, lr)

        y_train_pred = self.calculate_out(phi, self.weights)

        abs_res_error = self.calc_abs_res_error(train_Y, y_train_pred)

        return y_train_pred, abs_res_error


    def test(self, test_X, test_Y):
        """
        Forward pass
        """
        test_X = test_X.reshape(-1, 1)
        test_Y = test_Y.reshape(-1, 1)

        phi_test = self.calculate_phi(test_X)

        y_pred = self.calculate_out(phi_test, self.weights)

        abs_res_error = self.calc_abs_res_error(test_Y, y_pred)

        return y_pred, abs_res_error


    def transfer_function(self, x, mu_i, sigma_i):
        """
        Calculate the output of a Gaussian RBF node i
        """
        numerator = - ( x - mu_i) ** 2
        denominator = 2 * (sigma_i ** 2)
        exp_term = numerator / denominator
        return np.exp(exp_term)


    def calculate_phi(self, train_X):
        """
        Calculate the output of the rbf nodes for every training sample
        """
        # Initialize the output of the RBF nodes
        phi = np.empty(shape=(train_X.shape[0], len(self.RBF_Layer)))

        # TODO: this can be optimized to not use 2 for loops, but pass it as a matrix
        for i, sample in enumerate(train_X):
            for j, node in enumerate(self.RBF_Layer):
                phi[i][j] = self.transfer_function(sample, node.mu, node.sigma)

        return phi


    def calculate_out(self, phi, w):
        """
        Calculate an approximation of the output function given the output
        of the RBF nodes and the weights of the hidden layer
        """
        f_out = np.dot(phi, w)
        return f_out


    def calc_abs_res_error(self, f, f_pred):
        """
        Calculate the absolute residual error (average absolute difference between network outputs(f_pred) and target values(f))
        """
        error = 0.
        for i in range(f.shape[0]):
            error += np.abs(f[i] - f_pred[i])

        return error / f.shape[0]


    def lstsq(self, phi, f, _):
        """
        Update the weights using batch learning and the least square solution
        i.e Obtain w which minimizes the system phi.T * f_pred = phi.T * f
        """
        N = f.shape[0]  # Number of data points
        n = phi.shape[1]  # Number of nodes

        #if (N > n):  # Number of data points should always be lower or equal to number of nodes
        #   print("System overdetermined. Cannot solve it using least squares.")
        #   exit()

        pseudo_inv_phi = np.dot(phi.T, phi)
        pseudo_inv_f = np.dot(phi.T, f)
        # NOTE: This is using a ready made function (Working correctly)
        w = np.linalg.lstsq(pseudo_inv_phi, pseudo_inv_f, rcond=None)[0]
        # NOTE: This is manually solving the equation (Currently(!) not working correctly)
        # pseudo_inv_phi = np.linalg.inv(pseudo_inv_phi)
        # w = np.dot(pseudo_inv_phi.T, pseudo_inv_f)
        return w


    def delta_rule(self, phi, f, lr):
        """
        Update the weights using sequential (online) learning and delta rule
        """
        # Initialize weights randomly
        w = np.random.rand(phi.shape[1]).reshape(-1, 1)

        for j in range(100):
            # Shuffle the data for sequential learning
            indices = list(range(phi.shape[0]))
            np.random.shuffle(indices)
            for i in indices:
                phi_w = np.dot(phi, w).reshape(-1, 1)
                target_error = f - phi_w
                error = np.dot(target_error.T, phi).T
                delta_w = lr * error
                w = w + delta_w

        return w
