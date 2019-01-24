import numpy as np
from matplotlib import pyplot as plt

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
            "test_data": None,
            "test_targets": None,
            "m_weights": 0.1,
            "sigma_weights": 0.05,
            "nodes": 1,
            "learn_method": 'perceptron',
            "error": 1.0
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        n_features = data.shape[1]
        self.train_data, self.train_targets = self.shape_input(data, targets, n_features)
        self.w = self.init_weights(n_features)


    def init_weights(self, n_features):
        """
        Initialize weight matrix Features x Neurons
        """
        w = np.random.normal(self.m_weights, self.sigma_weights, n_features + 1)  # + 1 is bias

        for j in range(self.nodes - 1):
            w_j = np.random.normal(self.m_weights, self.sigma_weights, n_features + 1)  # + 1 is bias
            w = np.vstack((w, w_j))
        w = w.reshape(-1,1)
        return w

    def shape_input(self, X, Y, n_features):
        """
        Add bias as input vector (feature) in the data and shuffle them
        """
        index_shuffle = np.random.permutation(X.shape[0])
        X = X[index_shuffle]
        Y = Y[index_shuffle]
        bias = - np.ones((X.shape[0], 1))  # put a minus in front
        X = np.hstack((X, bias))  # changed so bias is after (before it was beginning)
        return X, Y

    def train(self, verbose=False):
        """
        Train neural network
        """
        iteration = 0
        self.int_w = {}
        self.error_history = []
        while iteration < self.epochs:
            #take a random batch or sequential batch?
            data = self.train_data
            targets = self.train_targets

            #predict (compute product of X * w)
            self.sum = np.dot(data, self.w)  # Y_pred: NxNeurons  "the output will be an NxNeurons matrix of sum for each neuron for each input vector"

            # compute delta_w
            delta_w = self.learn(data, targets)

            if self.error <= self.epsilon:
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

    def test():
        """
        Test trained ANN
        """
        pass

    def learn(self, data, targets):
        function = self.learning_method[self.learn_method]
        return function(data, targets)

    def perceptron(self, data, targets):
        """
        Perceptron learning
        """
        #pass result through activation function
        self.predictions = self.activation_function()
        diff = self.predictions - targets
        X = np.transpose(data)
        delta_w = self.learning_rate * np.dot(X, diff)

        # compute error
        self.error = self.missclass_error()
        self.error_history.append(self.error)

        return delta_w


    def delta_rule(self, data, targets):
        """
        Delta rule for computing delta w
        """
        diff = self.sum - targets
        X = np.transpose(data)
        delta_w = self.learning_rate * np.dot(X, diff)
        self.predictions = self.activation_function()

        # compute error
        self.error = np.mean(diff**2) #mse
        self.error_history.append(self.error)

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

    def missclass_error(self):
        """
        Calculate percentage of missclassification
        """
        miss = len(np.where(self.predictions != self.train_targets)[0])
        return float(miss/len(self.train_targets))

    def plot_decision_boundary_sequence(self, scatter = True):
        """
        Plot data as classified from the NN and the sequence
        of decision boundaries of the weights
        """
        fig, ax = plt.subplots()
        classA_ind = np.where(self.predictions > 0)[0]
        classB_ind = np.where(self.predictions <= 0)[0]

        classA_x1 = [self.train_data[:,0][i] for i in classA_ind]
        classA_x2 = [self.train_data[:,1][i] for i in classA_ind]
        classB_x1 = [self.train_data[:,0][i] for i in classB_ind]
        classB_x2 = [self.train_data[:,1][i] for i in classB_ind]

        # decision_boundary
        x1 = self.train_data[:, 0]
        for i in range(len(self.int_w)):
            part1 = self.int_w[i][0] / self.int_w[i][1]
            part2 = self.int_w[i][2] / self.int_w[i][1]
            x2 = np.array([- part1 * x + part2 for x in x1])
            ax.plot(x1, x2, 'b', alpha=float(i + 1) / (len(self.int_w) + 1))

        part1 = self.w[0] / self.w[1]
        part2 = self.w[2] / self.w[1]
        x2 = np.array([- part1 * x + part2 for x in x1])
        ax.plot(x1, x2, 'r')

        if scatter:
            ax.scatter(classA_x1, classA_x2, color='cyan', alpha=0.7, s=7)
            ax.scatter(classB_x1, classB_x2, color='purple', alpha=0.7, s=7)

        plt.show()
        plt.close()
        return

    def plot_decision_boundary(self, scatter = True, ann_list = None):
        """
        Plot data as classified from the NN and the decision boundary of the weights
        """
        fig, ax = plt.subplots()
        classA_ind = np.where(self.predictions > 0)[0]
        classB_ind = np.where(self.predictions <= 0)[0]

        classA_x1 = [self.train_data[:,0][i] for i in classA_ind]
        classA_x2 = [self.train_data[:,1][i] for i in classA_ind]
        classB_x1 = [self.train_data[:,0][i] for i in classB_ind]
        classB_x2 = [self.train_data[:,1][i] for i in classB_ind]

        # decision_boundary
        x1 = self.train_data[:, 0]
        #x1 = np.arange(np.min(self.train_data[:, 0]), np.max(self.train_data[:, 0]), 0.1)
        part1 = self.w[0] / self.w[1]
        part2 = self.w[2] / self.w[1]
        x2 = np.array([- part1 * x + part2 for x in x1])
        ax.plot(x1, x2, '--', alpha=0.5, label = self.learn_method)

        if ann_list:
            for ann in ann_list:
                part1 = ann.w[0] / ann.w[1]
                part2 = ann.w[2] / ann.w[1]
                x2 = np.array([- part1 * x - part2 for x in x1])
                ax.plot(x1, x2, '--', alpha=0.5, label=ann.learn_method)

        if scatter:
            ax.scatter(classA_x1, classA_x2, color='cyan', alpha=0.7, s=7)
            ax.scatter(classB_x1, classB_x2, color='purple', alpha=0.7, s=7)

        #ax.set_xlim(np.min(x1) - 0.1, np.max(x1) + 0.1)
        #ax.set_ylim(np.min(self.train_data[:, 1]) - 0.1, np.max(self.train_data[:, 1]) + 0.1)
        ax.legend(frameon=False)
        ax.set_xlabel('$x_1$', fontsize=18)
        ax.set_ylabel('$x_2$', fontsize=18)
        plt.savefig('3_1_2_1.eps')
        plt.show()
        plt.close()
        return

    def plot_error_history(self):
        """
        Plot the history of the error (show how quickly the NN converges)
        """
        x_axis = range(1, len(self.error_history) + 1)
        y_axis = self.error_history
        plt.scatter(x_axis, y_axis, color='purple', alpha=0.7)
        plt.show()

    def print_info(self, iteration, error):
        print('Iter: {}'.format(iteration))
        print('Train Error: {}'.format(error))
        print('Weights: {}'.format(self.w))
        print('\n')
