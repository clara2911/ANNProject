"""
This file contains the ANN class, which includes all the machine learning of a selected ANN

Authors: Clara Tump, Kostis SZ and Romina Azz
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FormatStrFormatter

class ANN:

    def __init__(self, data, targets, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """
        self.activation_functions = {
            'step': self.step
        }
        self.learning_method = {
            'perceptron': self.perceptron,
            'delta_rule': self.delta_rule
        }
        var_defaults = {
            "learning_rate": 0.5,
            "batch_size": 1,
            "theta": 0,
            "epsilon": 0.0,
            "epochs": 10,
            "act_fun": 'step',
            "m_weights": 0.1,
            "sigma_weights": 0.05,
            "nodes": 1,
            "learn_method": 'perceptron',
            "error": 1.0,
            "bias": -1
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.n_features = data.shape[1]
        self.train_data, self.train_targets = data, targets
        self.w = self.init_weights()
        self.error_history = []


    def init_weights(self):
        """
        Initialize weight matrix Features x Neurons
        """
        w = np.random.normal(self.m_weights, self.sigma_weights, self.n_features)

        if (self.nodes == 1):
            return w.reshape(-1,1)
        else:
            for j in range(self.nodes - 1):
                w_j = np.random.normal(self.m_weights, self.sigma_weights, self.n_features)
                w = np.vstack((w, w_j))
        return w.T

    def train(self, method, verbose=False):
        """
        Main train function
        """
        if (method == 'batch'):
            return self.train_batch(verbose)
        else:
            return self.train_sequential(verbose)

    def train_batch(self, verbose):
        """
        Train neural network
        We see all the data and sum all the deltas in order to make a unique update per epoch.
        """
        iteration = 0
        self.int_w = {}
        self.error_history = []
        while True:
            #take a random batch or sequential batch?
            data = self.train_data
            targets = self.train_targets
            #predict (compute product of X * w)
            self.sum = np.dot(data, self.w)

            # compute delta_w
            delta_w = self.learn(data, targets)
            self.error_history.append(self.error)

            if self.error <= self.epsilon or iteration > self.epochs:
                if (verbose):
                    self.print_info(iteration, self.error)
                break

            # This comment is by Francesco
            #update
            self.w = self.w - delta_w
            self.int_w[iteration] = self.w

            # maybe even compute MSE?
            if (verbose):
                self.print_info(iteration, self.error)
            iteration += 1

    def train_sequential(self, verbose):
        """
        Train neural network using sequential method.
        In each epoch we update the weights each time we see a sample.
        """
        iteration = 0
        self.int_w = {}
        self.error_history = []
        # take a random batch or sequential batch?
        data = self.train_data
        targets = self.train_targets
        self.predictions = np.empty(len(targets))
        while iteration < self.epochs:

            for nd, d in enumerate(data):
                #predict (compute product of X * w)
                self.sum = np.dot(d, self.w)  # Y_pred: NxNeurons  "the output will be an NxNeurons matrix of sum for each neuron for each input vector"
                # compute delta_w
                delta_w = self.learn(d, targets[nd], num_data = nd)
                #update
                self.w = self.w - delta_w.reshape(-1,1)

            # compute error
            self.sum = np.dot(data, self.w)
            self.predictions = self.activation_function()
            if self.learning_method == 'perceptron':
                self.error = self.missclass_error(self.predictions, targets)
            else:
                self.error = np.mean((self.sum - targets) ** 2)  # mse

            self.error_history.append(self.error)
            if self.error <= self.epsilon:
                if (verbose):
                    self.print_info(iteration, self.error)
                break

            self.int_w[iteration] = self.w

            if (verbose):
                self.print_info(iteration, self.error)
            iteration += 1

    def test(self, test_data, test_targets):
        """
        Test trained ANN
        """
        test_predictions = np.dot(test_data, self.w)
        self.sum = test_predictions
        targets_pred = self.activation_function()

        error = self.missclass_error(targets_pred, test_targets)

        print('Test Error: ', error)
        return error


    def learn(self, data, targets, num_data=None):
        function = self.learning_method[self.learn_method]
        return function(data, targets, num_data=num_data)

    def perceptron(self, data, targets, num_data=None):
        """
        Perceptron learning
        """
        #pass result through activation function
        X = np.transpose(data)
        if num_data==None:
            self.predictions = self.activation_function()
            diff = self.predictions - targets
            delta_w = self.learning_rate * np.dot(X, diff)
            # compute error
            self.error = self.missclass_error(self.predictions, targets)
        else:
            self.predictions[num_data] = self.activation_function()
            diff = self.predictions[num_data] - targets
            delta_w = self.learning_rate * np.multiply( X , diff)
        return delta_w


    def delta_rule(self, data, targets, num_data=None):
        """
        Delta rule for computing delta w
        """
        X = np.transpose(data)

        if num_data == None:
            self.predictions = self.activation_function()
            diff = self.sum - targets
            delta_w = self.learning_rate * np.dot(X, diff)
            # compute error
            self.error = np.mean(diff ** 2)  # mse
        else:
            self.predictions[num_data] = self.activation_function()
            diff = self.sum - targets
            delta_w = self.learning_rate * np.multiply(X, diff)
        return delta_w


    def step(self):
        """
        Set predictions to 1 or -1 depending on threshold theta
        """
        Y_threshed = np.where(self.sum > self.theta, 1, -1)
        return Y_threshed

    def activation_function(self):
        """
        Pass predictions through an activation function
        """
        function = self.activation_functions[self.act_fun]
        return function()

    def missclass_error(self, predictions, targets):
        """
        Calculate percentage of missclassification
        """
        miss = len(np.where(predictions != targets)[0])
        return float(miss/len(targets))

    def mse(self, predictions, targets):
        print("shape predictions: ", predictions.shape)
        print('shape targets:  ', targets.shape)
        mse = mean_squared_error(predictions, targets)
        return mse

    def plot_decision_boundary(self,
                               scatter = True, # scatter data points: True/False
                               ann_list = None, # list of different models to compare
                               data=None,
                               plot_intermediate=False, # plot boundary after every epoch True/False
                               title=None, # title for plot
                               line_titles=None,
                               data_coloring=None, # color data points as targets or predictions
                               origin_grid = False): # plot x=0 and y=0 grid
        """
        Plot data as classified from the NN and the decision boundary of the weights
        """

        fig, ax = plt.subplots()
        if data_coloring is not None:
            classA_ind = np.where(data_coloring > 0)[0]
            classB_ind = np.where(data_coloring <= 0)[0]
        else:
            classA_ind = np.where(self.predictions > 0)[0]
            classB_ind = np.where(self.predictions <= 0)[0]
        classA_x1 = [data[:,0][i] for i in classA_ind]
        classA_x2 = [data[:,1][i] for i in classA_ind]
        classB_x1 = [data[:,0][i] for i in classB_ind]
        classB_x2 = [data[:,1][i] for i in classB_ind]


        x1 = data[:, 0]
        # plot the decision boundary after each iteration
        if plot_intermediate:
            for i in range(len(self.int_w)):
                part1 = self.int_w[i][0] / self.int_w[i][1]
                part2 = self.int_w[i][2] / self.int_w[i][1]
                x2 = np.array([- part1 * x - self.bias*part2 for x in x1])
                ax.plot(x1, x2, 'b', alpha=0.2*float(i + 1) /(len(self.int_w) + 1))
        #plot final decision boundary
        part1 = self.w[0] / self.w[1]
        part2 = self.w[2] / self.w[1]
        x2 = np.array([- part1 * x - self.bias*part2 for x in x1])
        ax.plot(x1, x2, alpha=0.5, color='red', linewidth=3, label=self.learning_method)

        # if you want to plot multiple ann's and compare them
        if ann_list:
            for ann in ann_list:
                part1 = ann.w[0] / ann.w[1]
                part2 = ann.w[2] / ann.w[1]
                x2 = np.array([- part1 * x - self.bias*part2 for x in x1])
                ax.plot(x1, x2, '--', alpha=0.5, label=ann.learn_method)

        # plot data points
        if scatter:
            ax.scatter(classA_x1, classA_x2, color='cyan', alpha=0.7, s=7)
            ax.scatter(classB_x1, classB_x2, color='purple', alpha=0.7, s=7)
        ax.set_xlim(np.min(x1) - 0.1, np.max(x1) + 0.1)
        ax.set_ylim(np.min(self.train_data[:, 1]) - 0.1, np.max(self.train_data[:, 1]) + 0.1)

        #plot origin grid
        if origin_grid:
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')

        if title:
            ax.set_title(title)
        custom_lines = [Line2D([0], [0], color='b'), Line2D([0], [0], color='r')]
        ax.legend(custom_lines, [line_titles[0], line_titles[1]], frameon=False, loc='upper right')
        # ax.legend(frameon=False)
        ax.set_xlabel('$x_1$', fontsize=18)
        ax.set_ylabel('$x_2$', fontsize=18)
        plt.savefig('../figures/non_separable_delta_perc.eps')
        plt.show()
        plt.close()
        return

    def plot_decision_boundary_general(self, scatter = True, ann_list = None, data=None, targets=None, title=None):
        """
        Plot data as classified from the NN and the decision boundary of the weights
        """
        fig, ax = plt.subplots()
        classA_ind = np.where(self.predictions > 0)[0]
        classB_ind = np.where(self.predictions <= 0)[0]

        classA_x1 = np.array([data[i,0] for i in classA_ind])
        classA_x2 = np.array([data[i,1] for i in classA_ind])
        classB_x1 = np.array([data[i,0] for i in classB_ind])
        classB_x2 = np.array([data[i,1] for i in classB_ind])

        # decision_boundary
        x1 = data[:, 0]
        part1 = self.w[0] / self.w[1]
        part2 = self.w[2] / self.w[1]
        x2 = np.array([- part1 * x - self.bias*part2 for x in x1])
        ax.plot(x1, x2, '--', alpha=0.5, label = self.learn_method)

        if ann_list:
            for ann in ann_list:
                part1 = ann.w[0] / ann.w[1]
                part2 = ann.w[2] / ann.w[1]
                x2 = np.array([- part1 * x - self.bias*part2 for x in x1])
                ax.plot(x1, x2, '--', alpha=0.5, label=ann.learn_method)

        if scatter:
            ax.scatter(classA_x1, classA_x2, color='cyan', alpha=0.7, s=7)
            ax.scatter(classB_x1, classB_x2, color='purple', alpha=0.7, s=7)

        #ax.set_xlim(np.min(x1) - 0.1, np.max(x1) + 0.1)
        #ax.set_ylim(np.min(self.train_data[:, 1]) - 0.1, np.max(self.train_data[:, 1]) + 0.1)
        ax.legend(frameon=False)
        ax.set_xlabel('$x_1$', fontsize=18)
        ax.set_ylabel('$x_2$', fontsize=18)
        ax.set_title(title)
        words_title = title.split(' ')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.savefig('../figures/3_1_3_{}_{}.eps'.format(words_title[0], words_title[5]))
        plt.show()
        plt.close()
        print('Final error: {}'.format(self.error))
        return

    def plot_error_history(self):
        """
        Plot the history of the error (show how quickly the NN converges)
        """
        x_axis = range(1, len(self.error_history) + 1)
        y_axis = self.error_history
        plt.scatter(x_axis, y_axis, color='red', alpha=0.7)
        plt.show()

    def print_info(self, iteration, error):
        print('Iter: {}'.format(iteration))
        print('Train Error: {}'.format(error))
        print('Weights: {}'.format(self.w))
        print('\n')
