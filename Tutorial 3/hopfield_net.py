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

    def recall(self, recall_set):
        """ """
        num_recall = recall_set.shape[0]
        recalled_patterns = np.zeros((num_recall, self.num_feats))
        for i in range(num_recall):
            x = recall_set[i,:]
            x_updated = np.zeros(x.shape)
            for j, w in enumerate(self.W):
                x_updated[j] = np.sign(w.dot(x))
            recalled_patterns[i,:] = x_updated
        return recalled_patterns

    def batch_train(self):
        W = np.zeros((self.num_feats, self.num_feats))
        for x in self.train_samples:
            W += np.outer(x, x) - np.eye(self.num_feats)
        self.W = W

    def init_weights(self):
        self.W = np.zeros((self.num_feats, self.num_feats))


if __name__ == '__main__':
    """ Test for the hopfield network """

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
