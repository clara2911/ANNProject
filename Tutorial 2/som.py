#!/usr/bin/env python
"""
Main file for assignment 4.1
Topological Ordering of Animal Species

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""

import numpy as np

class Som:

    def __init__(self, train_examples, **kwargs):
        """
        Initialize algorithm with data and parameters
        """
        var_defaults = {
            "epochs" : 20,
            "step_size" : 0.2,
            "num_nodes" : 100
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.num_examples = train_examples.shape[1]
        self.num_feats = train_examples.shape[0]
        self.train_examples = train_examples

    def train(self):
        """
        Train the SOM algorithm
        """
        weights = self.init_weights(self.num_nodes, self.num_feats)
        # for i in range(self.epochs):
        #     for j in range(self.num_examples):
            #     self.similarity_calc(self.train_examples[j], each_node)
            #     winner_index = self.find_winner(data_vec, weights)
            #     neighbors = self.neighborhood(winner_index)
            #     self.update_weights(neighbors)
            # self.update_neighbor_function(i)

    def init_weights(self, p, q):
        """
        initialize weight matrix of size pxq with random numbers
        between zero and one.
        """
        # Todo: generalize so you can pass min, max instead of 0,1
        weights = np.random.rand(p, q)
        print("initialized weights. Shape: ", weights.shape)
        return weights



    def similarity_calc(self, input, output_weights):
        """
        Calculate the similarity between the input pattern and the
        weights arriving at each output node.
        """
        pass

    def _find_winner(self, similarities):
        """
        Find the most similar node; often referred to as the winner.
        """
        # return the index of the winning node
        pass

    def _neighborhood(self, winner, output_grid):
        """
        Select a set of output nodes which are located close to the
        winner in the output grid. This is called the neighbourhood.
        """
        pass

    def _update_weights(self, weights):
        """
        Update the weights of all nodes in the neighbourhood such that
        their weights are moved closer to the input pattern.
        """
        pass

    def _update_neighbor_func(self, i):
        """
        You should start with a large neighbourhood and gradually make
        it smaller. Make the size of the neighbourhood depend on the
        epoch loop variable so that you start with a neighbourhood of
        about 50 and end up close to one or zero.
        i = the current epoch
        """

        pass