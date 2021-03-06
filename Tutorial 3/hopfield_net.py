#!/usr/bin/env python
"""
hopfield_net.py
Implements a Hopfield network

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np
from plot import show_tested
from plot import show_trained


class HopfieldNet:

    def __init__(self, train_samples):
        """
        Initialize algorithm with data and parameters
        """
        self.num_examples = train_samples.shape[0]
        self.num_feats = train_samples.shape[1]
        self.train_samples = train_samples
        self.W = self._init_weights()

    def _init_weights(self):
        """
        Currently weights are initialized to zero.
        W is a matrix of weights which indicates the weight associated between
        two nodes. Ex: wij (weight in row=i and col=j) indicates the weight
        between node i and j. Given this W is symmetric, with diagonal = 0.
        """
        return np.zeros((self.num_feats, self.num_feats))

    def batch_train(self):
        """
        We train with all patterns at the same time.
        The final weights are the result of summing the outer product
        of the patterns learned. We subtract an identity to this matrix bacause
         we have to keep diagonal = 0, given that we cannot consider the product
         of a node with itself.
        *** this is the method provided by clara's video and the assignment
        """
        for x in self.train_samples:
            self.W += np.outer(x, x) - np.eye(self.num_feats)

    def sparse_train(self, rho):
        """
        We train with all patterns at the same time.
        The final weights are the result of summing the outer product
        of the patterns learned. We subtract an identity to this matrix bacause
         we have to keep diagonal = 0, given that we cannot consider the product
         of a node with itself.
        *** this is the method provided by clara's video and the assignment
        """
        rho_vec = rho * np.ones(self.num_feats)
        for x in self.train_samples:
            self.W += np.outer(x - rho_vec, x - rho_vec) - np.eye(self.num_feats)


    def recall(self, test_set, epochs=10, threshold=0.):
        """
        function to reconstruct a learned pattern:
        reconstructed pattern is equal to the product of the provided
        recall sample with the weights.
        """
        y = test_set
        energy = []
        for _ in range(epochs):
            y = np.dot(y, self.W)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if y[i,j] < 0:
                        y[i,j] = -1
                    else:
                        y[i,j] = 1
            energy += [self.energy(y)]
        return y, energy

    def recall_01s(self, test_set, epochs=10, threshold=0.):
        """
        function to reconstruct a learned pattern:
        reconstructed pattern is equal to the product of the provided
        recall sample with the weights.
        Used for 3.6 where we use {0,1} instead of {-1,1}
        """
        y = test_set
        for _ in range(epochs):
            y = np.dot(y, self.W )- threshold
            y = 0.5*np.ones(y.shape) + 0.5*self._sign(y)

        return y

    def _sign(self, y):
        """
        unlike the numpy sign function this returns 1 if the input is 0.
        """
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i, j] < 0:
                    y[i, j] = -1
                else:
                    y[i, j] = 1
        return y

    def sequential_recall(self, recall_set, epochs=1000, threshold=0., plot_at_100=False):
        """
        function to reconstruct a learned pattern:
        reconstructed pattern is equal to the product of the provided
        recall sample with the weights.

        recall_set: each row contains a pattern that is going to be use to
        recall a learned pattern from it.

        threshold: For this assignment, default is 0.
        """
        num_recall = recall_set.shape[0]
        recalled_patterns = np.zeros((num_recall, self.num_feats))

        energy_evol = {}
        # Iterate through patterns to recall
        for i in range(num_recall):
            energy_evol[i] = []
            pattern_i = recall_set[i, :]
            recalled_patterns[i, :] = pattern_i
            # Iterate through epochs
            for epoch in range(epochs):
                # Pick a random unit each time
                rand_pick = np.random.choice(range(len(pattern_i)))
                sign = np.sign(self.W[rand_pick, :].dot(pattern_i) - threshold)
                if sign == 0.0:
                    sign = 1.0
                recalled_patterns[i, rand_pick] = sign
                energy_evol[i] += [self.energy(recalled_patterns[i, :])]

                # Plot evolution through time
                #if plot_at_100 and epoch % 100 == 0:
                #    show_tested(recall_set[i], recalled_patterns[i].reshape(32, 32), 32, 32)

        return recalled_patterns, energy_evol

    def sequential_recall_shuffle(self, recall_set, epochs=1000, threshold=0., plot_at_100=False):
        """
        function to reconstruct a learned pattern:
        reconstructed pattern is equal to the product of the provided
        recall sample with the weights.

        recall_set: each row contains a pattern that is going to be use to
        recall a learned pattern from it.

        threshold: For this assignment, default is 0.
        """
        num_recall = recall_set.shape[0]
        recalled_patterns = np.zeros((num_recall, self.num_feats))

        energy_evol = {}
        # Iterate through patterns to recall
        for i in range(num_recall):
            energy_evol[i] = []
            pattern_i = recall_set[i, :]
            random_index = np.arange(0,len(pattern_i))
            np.random.shuffle(random_index)
            # Iterate through epochs
            for epoch in range(epochs):
                # Pick a random unit each time
                for ind in random_index:
                    rand_pick = ind
                    sign = np.sign(self.W[rand_pick, :].dot(pattern_i) - threshold)
                    if sign == 0.0:
                        sign = 1.0
                    pattern_i[rand_pick] = sign
                    energy_evol[i] += [self.energy(pattern_i)]
            recalled_patterns[i, :] = pattern_i

                # Plot evolution through time
                #if plot_at_100 and epoch % 100 == 0:
                #    show_tested(recall_set[i], recalled_patterns[i].reshape(32, 32), 32, 32)

        return recalled_patterns, energy_evol

    def sequential_recall_01s(self, recall_set, epochs=1000, threshold=0., plot_at_100=False):
        """
        used for 3.6, where the inputs are binary {0,1} instead of {-1,1}
        """
        num_recall = recall_set.shape[0]
        recalled_patterns = np.zeros((num_recall, self.num_feats))
        for i in range(num_recall):
            pattern_i = recall_set[i, :]
            choose = range(len(pattern_i))
            for _ in range(epochs):
                rand_pick = np.random.choice(choose)
                recalled_patterns[i, rand_pick] = 0.5 + 0.5*np.sign(self.W[rand_pick, :].dot(pattern_i) - threshold)
        return recalled_patterns

    def energy(self, pattern, threshold = 0):
        """ Function of energy, this allows to keep in track the convergence
        of the algorithm. It should decrease with each step.
        For more information: https://www.doc.ic.ac.uk/~sd4215/hopfield.html
        """
        xxt = np.outer(pattern, pattern)
        E = -0.5*np.sum(self.W*xxt) + np.sum(threshold*pattern)
        return E
