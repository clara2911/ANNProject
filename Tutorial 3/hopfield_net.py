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
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.num_examples = train_examples.shape[1]

    def train(self):
        pass

    def recall():
        pass

    def batch_train():
        pass

    def seq_train():
        pass

    def show_pattern():
        # plot the pattern so that you can see the picture
        pass


