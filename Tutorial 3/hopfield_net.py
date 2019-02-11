#!/usr/bin/env python
"""
hopfield_net.py
Implements a Hopfield network

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np


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
        Synchronous update. Using all units to update at the same time.
        The final weights are the result of summing the outer product
        of the patterns learned. We subtract an identity to this matrix because
        we have to keep the diagonal = 0, given that we cannot consider the product
        of a node with itself.
        """
        for x in self.train_samples:
            self.W += np.outer(x, x) - np.eye(self.num_feats)

    def sequential_update(self):
        """
        Asynchronous update. Using only one unit, picked randomly, to update at a time.
        """
        pass


    def recall(self, recall_set, epochs=100, threshold=0.):
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
        for epoch in range(epochs):
            for i in range(num_recall):
                pattern_i = recall_set[i,:]
                for j, w in enumerate(self.W):
                    recalled_patterns[i, j] = np.sign(w.dot(pattern_i) - threshold)

        return recalled_patterns

    def energy(self, pattern, threshold = 0):
        """ Function of energy, this allows to keep in track the convergence
        of the algorithm. It should decrease with each step.
        For more information: https://www.doc.ic.ac.uk/~sd4215/hopfield.html
        """
        xxt = np.outer(pattern, pattern)
        E = -0.5*np.sum(self.W*xxt) + np.sum(threshold*pattern)
        return E
