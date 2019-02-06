#!/usr/bin/env python
"""
Implements the SOM algorithm
functions
init: initializes a som object with given train examples
train: trains the som algorithm using the train examples
apply: returns a position array with the ordering for the given examples
order: orders a given set of examples using the position array
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
            "num_nodes" : [10,10] # 1-D: eg [100] or 2-D eg [10,10]
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.num_examples = train_examples.shape[1]
        self.num_feats = train_examples.shape[0]
        self.train_examples = train_examples
        # Todo: think of something else instead of this ugly if/elif/else for 1-D and 2-D
        # Todo: Maybe recursion?
        if len(self.num_nodes) == 2:
            self.weights = self._init_weights(self.num_nodes[0], self.num_nodes[1], self.num_feats)
        elif len(self.num_nodes) == 1:
            self.weights = self._init_weights(self.num_nodes[0], self.num_feats)
        else:
            exit("use a 1D or 2D grid!")
        # Todo: I don't really know how to make a self.var and also save its initial value
        # Todo: but I don't think declaring both of these here is good practice
        self.neighborhood_size_init = self._init_neighborhood_size(self.weights.shape)
        self.neighborhood_size = self._init_neighborhood_size(self.weights.shape)

    def train(self):
        """
        Train the SOM algorithm
        """
        #Todo: shuffle data points (between epochs)?
        for i in range(self.epochs):
            self._compute_neighborhood_size(i)
            for j in range(self.num_examples):
                data_vec = self.train_examples[:,j]
                winner_index = self._find_winner(data_vec)
                neighborhood_indices = self._find_neighbors(winner_index)
                self._update_weights(data_vec, neighborhood_indices)

    def apply(self, apply_examples):
        """
        Returns an array with the positions of each given example in apply_examples
        """
        num_examples = apply_examples.shape[-1]
        pos = np.zeros([num_examples, len(self.num_nodes)], dtype=int)
        for i in range(num_examples):
            data_vec = apply_examples[:, i]
            winner_index = self._find_winner(data_vec)
            pos[i] = winner_index
        return pos

    def order(self, pos, apply_examples, names):
        """
        returns the feature array apply_examples and the list of names sorted by the pos array
        """
        # Todo: generalize for multi-D
        info_tuples = []
        num_examples = apply_examples.shape[-1]
        for i in range(num_examples):
            info_tuples.append((pos[i], apply_examples[:,i], names[i]))
        sorted_pos = sorted(info_tuples, key=self._take_first)
        sorted_data = np.column_stack([x[-2] for x in sorted_pos])
        sorted_names = np.array([x[-1] for x in sorted_pos], dtype=str)
        return sorted_data, sorted_names

    def _take_first(self, elem):
        """
        method needed for sorting by the first element of a tuple
        """
        return elem[0]


    def _init_weights(self, *args):
        """
        initialize weight matrix of size pxq with random numbers
        between zero and one.
        """
        weights = np.random.rand(*args)
        return weights


    def _init_neighborhood_size(self, shape_weights):
        """
        initializes the neighborhood size. Neighborhood will be (winner - nbh_size, winner + nbh_size)
        Initial value: a quarter of the number of output nodes
        """
        neighborhood_size = np.zeros(len(shape_weights)-1, dtype=int)
        for i in range(len(shape_weights[:-1])):
            dim_len = shape_weights[i]
            neighborhood_size[i] = dim_len / 4
        return neighborhood_size


    def _find_winner(self, train_example):
        """
        Find the most similar node; often referred to as the winner.
        This is the node with the minimum index to the train_example
        Return its index
        """
        # if 1 dimension you have to iterate over the first dim of the vector
        # if grid, you have to iterate over the first dim, and then iterate over the second dim
        # and add all distances to dists
        # Todo: think of something else instead of this ugly if/elif/else for 1-D and 2-D
        # Todo: Maybe recursion?
        if len(self.num_nodes) == 1:
            dists = np.array([self._dist(train_example, weight_i) for weight_i in self.weights])
        elif len(self.num_nodes) == 2:
            dists = np.array([[self._dist(train_example, weight_i) for weight_i in grid_vec] for grid_vec in self.weights])
        else:
            exit("Use a 1D or 2D grid!")
        return np.array(np.unravel_index(dists.argmin(), dists.shape)) # does the same as argmin(dists), but also works in 2D


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
        min_neighbor = winner-self.neighborhood_size[0]
        # index shouldn't go below 0
        min_neighbor[min_neighbor < 0] = 0
        max_neighbor = winner+self.neighborhood_size[0]
        # index shouldn't go above num_nodes-1
        # Todo: this breaks if the grid is not rectangular
        max_neighbor[max_neighbor > self.num_nodes[0]] = self.num_nodes[0]-1
        # Todo: think of something else instead of this ugly if/elif/else for 1-D and 2-D
        neighbor_indices = []
        if len(self.num_nodes) == 1:
            for dim1 in range(min_neighbor[0], max_neighbor[0]):
                neighbor_indices.append(np.array([dim1]))
        elif len(self.num_nodes) == 2:
            for dim1 in range(min_neighbor[0], max_neighbor[0]):
                for dim2 in range(min_neighbor[1], max_neighbor[1]):
                    neighbor_indices.append(np.array([dim1, dim2]))
        else:
            exit("Use a 1D or 2D grid!")
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
            step_per_it = self.neighborhood_size_init[i] / self.epochs
            self.neighborhood_size[i] = int(self.neighborhood_size_init[i] - step_per_it*iteration)