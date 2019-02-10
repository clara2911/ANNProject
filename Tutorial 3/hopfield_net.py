#!/usr/bin/env python
"""
hopfield_net.py
Implements a Hopfield network

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

class HopfieldNet:

    def __init__(self, train_examples, **kwargs):
        """
        Initialize algorithm with data and parameters
        """
        var_defaults = {
            "epochs" : 20,
            "neurons": 10,
            "learn_method": 'classic'
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.num_examples = train_examples.shape[0]
        self.num_feats = train_examples.shape[1]
        self.W = np.zeros((self.num_feats, self.num_feats))
        self.train_samples = train_examples
        self.recalled_patterns = np.zeros(train_examples.shape)

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
        for i in range(num_recall):
            x = recall_set[i,:]
            x_updated = np.zeros(x.shape)
            for j, w in enumerate(self.W):
                x_updated[j] = np.sign(w.dot(x) - threshold)
            recalled_patterns[i,:] = x_updated
        return recalled_patterns

    def batch_train(self):
        """
        We train with all patterns at the same time.
        The final weights are the result of summing the outer product
        of the patterns learned. We substract an identity to this matrix bacause
         we have to keep diagonal = 0, given that we cannot consider the product
         of a node with itself.
        *** this is the method provided by clara's video and the assignment
        """

        W = np.zeros((self.num_feats, self.num_feats))
        for x in self.train_samples:
            W += np.outer(x, x) - np.eye(self.num_feats)
        self.W = W

    def init_weights(self):
        """
        For the moment we initialize the weights at zero.
        W is a matrix of weights which indicates the weight associated between
        two nodes. Ex: wij (weight in row=i and col=j) indicates the weight
        between node i and j. Given this W is symmetric, with diagonal = 0.
        """
        self.W = np.zeros((self.num_feats, self.num_feats))

    def Energy(self, pattern, threshold = 0):
        """ Function of energy, this allows to keep in track the convergence
        of the algorithm. It should decrease with each step.
        For more information: https://www.doc.ic.ac.uk/~sd4215/hopfield.html
        """
        xxt = np.outer(pattern, pattern)
        E = -0.5*np.sum(self.W*xxt) + np.sum(threshold*pattern)
        return E



if __name__ == '__main__':
    """ Test for the hopfield network """
    """
    This is just for testing the functions
    """

    x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
    x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
    x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
    train_set = np.vstack((x1, x2))
    train_set = np.vstack((train_set, x3))

    params = {
        "epochs": 20,
        "neurons": len(x1),
        "learn_method": 'classic'
    }

    Hop = HopfieldNet(train_set, **params)
    Hop.batch_train()
    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]
    recall_set = np.vstack((x1d, x2d))
    recall_set = np.vstack((recall_set, x3d))
    recalled_set = Hop.recall(recall_set)
    print("hola")
