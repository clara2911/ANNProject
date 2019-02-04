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

        self.weights = self._init_weights(self.num_nodes, self.num_feats)
        # Todo: I don't really know how to make a self.var and also save it's initial value
        # Todo: but I don't think declaring both of these here is good practice
        self.neighborhood_size_init = self._init_neighborhood_size(self.weights.shape)
        self.neighborhood_size = self._init_neighborhood_size(self.weights.shape)

    def train(self):
        """
        Train the SOM algorithm
        """
        #Todo: shuffle data points ?
        for i in range(self.epochs):
            print("Epoch: ", i)
            self._compute_neighborhood_size(i)
            for j in range(self.num_examples):
                data_vec = self.train_examples[:,j]
                winner_index = self._find_winner(data_vec)
                neighborhood_indices = self._find_neighbors(winner_index)
                self._update_weights(data_vec, neighborhood_indices)


    def _init_weights(self, p, q):
        # Todo: generalize the weights so it can also be a 2D grid
        """
        initialize weight matrix of size pxq with random numbers
        between zero and one.
        """
        weights = np.random.rand(p, q)
        print("initialized weights. Shape: ", weights.shape)
        return weights

    def _init_neighborhood_size(self, shape_weights):
        neighborhood_size = np.zeros(len(shape_weights)-1)
        for i in range(len(shape_weights[:-1])):
            dim_len = shape_weights[i]
            neighborhood_size[i] = dim_len / 2
        return neighborhood_size.astype(int)


    def _find_winner(self, train_example):
        """
        Find the most similar node; often referred to as the winner.
        This is the node with the minimum index to the train_example
        Return its index
        """
        dists = [self._dist(train_example, weight_i) for weight_i in self.weights]
        return np.argmin(dists)

    def _dist(self, vec1, vec2):
        """
        Calculate the distance between two vectors.
        Note: square root is left out so it's not real Euclidean dist.
        """
        diff = vec1 - vec2
        distance = np.dot(np.transpose(diff),diff)
        return distance

    def _find_neighbors(self, winner):
        """
        Select a set of output nodes which are located close to the
        winner in the output grid. This is called the neighbourhood.
        """
        # Todo: generalize to more than 1D
        # index shouldn't go below 0 or above num_nodes-1
        min_neighbor = max(0,winner-self.neighborhood_size[0])
        max_neighbor = min(winner+self.neighborhood_size[0],self.num_nodes-1)
        neighbor_indices = list(range(min_neighbor, max_neighbor))
        return neighbor_indices

    def _update_weights(self, data_vec, neighborhood_indices):
        """
        Update the weights of all nodes in the neighbourhood such that
        their weights are moved closer to the input pattern.
        """
        for index in neighborhood_indices:
            update = self.step_size * (data_vec - self.weights[index])
            self.weights[index] += update

    def _compute_neighborhood_size(self, iteration):
        """
        You should start with a large neighbourhood and gradually make
        it smaller. Make the size of the neighbourhood depend on the
        epoch loop variable so that you start with a neighbourhood of
        about 50 and end up close to one or zero.
        iteration = the current epoch
        """
        # neighborhood size is a numpy array with 1 int per node-grid dimension
        for i in range(len(self.neighborhood_size)):
            size_range = self.neighborhood_size_init[i] - 1
            step_per_it = size_range / self.epochs
            self.neighborhood_size[i] = int(self.neighborhood_size_init[i] - step_per_it*iteration)