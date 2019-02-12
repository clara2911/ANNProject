#!/usr/bin/env python
"""
hopfield_net.py
Implements a Hopfield network

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

class HopfieldNet:

    def __init__(self, train_samples, **kwargs):
        """
        Initialize algorithm with data and parameters
        """
        var_defaults = {
            "epochs" : 100,
            "neurons": 10,
            "learn_method": 'classic'
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.num_examples = train_samples.shape[0]
        self.num_feats = train_samples.shape[1]
        self.W = self._init_weights()
        self.train_samples = train_samples
        self.recalled_patterns = np.zeros(train_samples.shape)

    def recall(self, recall_set, threshold = 0):
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
        for epoch in range(self.epochs):
            for i in range(num_recall):
                pattern_i = recall_set[i,:]
                for j, w in enumerate(self.W):
                    recalled_patterns[i, j] = np.sign(w.dot(pattern_i) - threshold)
        return recalled_patterns

    def sequential_recall(self, recall_set, threshold = 0):
        """
        function to reconstruct a learned pattern:
        reconstructed pattern is equal to the product of the provided
        recall sample with the weights.

        recall_set: each row contains a pattern that is going to be use to
        recall a learned pattern from it.

        threshold: For this assignment, default is 0.
        """
        num_recall = recall_set.shape[0]
        energy_evol = {}
        recalled_patterns = np.zeros((num_recall, self.num_feats))
        for i in range(num_recall):
            energy_evol[i] = []
            pattern_i = recall_set[i, :]
            for epoch in range(self.epochs):
                print(epoch)
                rand_pick = np.random.choice(range(len(pattern_i)))
                pattern_i[rand_pick] = np.sign(self.W[rand_pick,:].dot(pattern_i) - threshold)
                energy_evol[i] += [self.energy(pattern_i)]
            recalled_patterns[i, :] = pattern_i
        return recalled_patterns, energy_evol

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

    def _init_weights(self):
        """
        For the moment we initialize the weights at zero.
        W is a matrix of weights which indicates the weight associated between
        two nodes. Ex: wij (weight in row=i and col=j) indicates the weight
        between node i and j. Given this W is symmetric, with diagonal = 0.
        """
        return np.zeros((self.num_feats, self.num_feats))

    def energy(self, pattern, threshold = 0):
        """ Function of energy, this allows to keep in track the convergence
        of the algorithm. It should decrease with each step.
        For more information: https://www.doc.ic.ac.uk/~sd4215/hopfield.html
        """
        xxt = np.outer(pattern, pattern)
        E = -0.5*np.sum(self.W*xxt) + np.sum(threshold*pattern)
        return E
