#!/usr/bin/env python
"""
File: vec_quant.py
RBF initialization using Competitive Learning (Vector Quantisation)

Authors: Kostis SZ, Romina Ariazza and Clara Tump

----------------------------------------------------------------

Algorithm vector quantization:

you have a sin(2x) function you want to predict
You have 10 RBF nodes put randomly/equally in the x space.
CL:
for i iterations:
    pick a random data point from the sin(2x) vector
    compute the nearest RBF node (the winner)
    Update the winner by moving it closer to the data point
for each RBF node:
    calculate the sigma

Improvements (?):
- update the neighbors too
"""

import numpy as np
from collections import defaultdict

class VecQuantization:

    def __init__(self, rbf_nodes, iterations=100, step_size=0.1, neighbor_bool = True):
        self.rbf_nodes = rbf_nodes
        self.iterations = iterations
        self.step_size = step_size
        self.neighbor_bool = neighbor_bool

        self.num_nodes = len(rbf_nodes)
        self.neighborhood_size_init = self._init_neighborhood_size(self.num_nodes)
        self.neighborhood_size = self._init_neighborhood_size(self.num_nodes)


    def move_RBF(self, data_points):
        """
        main method.
        Moves the RBF nodes using the competitive learning Vector Quantisation algorithm
        so that they are initialized in a smarter way than just random/distributed evenly over the domain
        """
        for i in range(self.iterations):
            self._compute_neighborhood_size(i)
            data_point = self._pick_data_point(data_points)
            winner = self._find_winner(data_point)
            if self.neighbor_bool:
                neighbors = self._find_neighbors(winner)
            else:
                neighbors = np.array([winner])
            self._update_mu(data_point, neighbors)
        self._calc_sigmas(data_points)
        return self.rbf_nodes

    def _init_neighborhood_size(self, num_nodes):
        """
        initializes the neighborhood size. Neighborhood will be (winner - nbh_size, winner + nbh_size)
        Initial value: a quarter of the number of output nodes
        """
        return np.array([int(num_nodes / 4)])

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
            step_per_it = self.neighborhood_size_init[i] / self.iterations
            self.neighborhood_size[i] = int(self.neighborhood_size_init[i] - step_per_it*iteration)

    def _pick_data_point(self, data_points):
        """
        a training vector is randomly selected from the data.
        """
        data_point = np.random.choice(data_points)
        return data_point

    def _find_winner(self, data_point):
        """
        Compute the closest RBF unit to the picked data point (usually called the winning unit)
        """
        dists = np.array([self._dist(data_point, rbf_node_i.mu) for rbf_node_i in self.rbf_nodes])
        winner = np.array(np.unravel_index(dists.argmin(), dists.shape))
        return winner[0]

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
        max_neighbor = winner+self.neighborhood_size[0]
        neighbor_indices = self.linear_neighbors(min_neighbor, max_neighbor)
        return neighbor_indices

    def linear_neighbors(self, min_neighbor, max_neighbor):
        # index shouldn't go below 0
        if min_neighbor < 0:
            min_neighbor = 0
        # Todo: this breaks if the grid is not rectangular
        # index shouldn't go above num_nodes-1
        if max_neighbor >= self.num_nodes:
            max_neighbor = self.num_nodes - 1
        neighbor_indices = np.array(range(min_neighbor, max_neighbor+1))
        return neighbor_indices


    def _update_mu(self, data_point, neighborhood_indices):
        """
        the winning RBF unit is updated, in such a way that it gets closer to the training vector.
        #TODO: Right now it is just the winner but we can define a neighborhood too
        """
        for index in neighborhood_indices:
            update = self.step_size * (data_point - self.rbf_nodes[index].mu)
            self.rbf_nodes[index].mu += update

    def _calc_sigmas(self, data_points):
        """
        make the adjustments to the RBF nodes' widths (sigmas) manually based on the distribution of data around
        the cluster centers found with this CL algorithm.
        """
        clusters = defaultdict(list)
        for data_point in data_points:
            winner = self._find_winner(data_point)
            clusters[winner].append(data_point)
        for rbf_node, clus_data_points in clusters.items():
            self.rbf_nodes[rbf_node].sigma = np.var(clus_data_points)*10+0.01



