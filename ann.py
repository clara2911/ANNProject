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
            "nodes": 1
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
        error = 1.
        self.int_w = {}
        self.error_history = []
        while(error > self.epsilon and iteration < self.epochs):
            #take a random batch or sequential batch?
            data = self.train_data
            targets = self.train_targets
            #predict (compute product of X * w)
            self.predictions = np.dot(data, self.w)  # Y_pred: NxNeurons  "the output will be an NxNeurons matrix of sum for each neuron for each input vector"
            #compute delta_w using the perceptron learning
            delta_w = self.perceptron(data, targets)
            # delta_w = self.delta_rule()
            #update
            self.w = self.w - delta_w
            #compute error
            error = self.missclass_error()
            self.error_history.append(error)
            # maybe even compute MSE?
            self.int_w[iteration] = self.w
            iteration += 1
            if (verbose):
                self.print_info(iteration, error)

    def test():
        """
        Test trained ANN
        """
        pass

    def perceptron(self, data, targets):
        """
        Perceptron learning
        """
        #pass result through activation function
        self.predictions = self.activation_function()
        delta_w = self.delta_rule(data, targets)
        return delta_w


    def delta_rule(self, data, targets):
        """
        Delta rule for computing delta w
        """
        diff = self.predictions - targets
        X = np.transpose(data)
        delta_w = self.learning_rate * np.dot(X, diff)
        return delta_w


    def step(self):
        """
        Set predictions to 1 or -1 depending on threshold theta
        """
        Y_threshed = np.where(self.predictions > self.theta, 1, -1)
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

    def plot_decision_boundary(self):
        """
        Plot data as classified from the NN and the decision boundary of the weights
        """
        classA_ind = np.where(self.predictions == 1)[0]
        classB_ind = np.where(self.predictions == -1)[0]

        classA_x1 = [self.train_data[:,0][i] for i in classA_ind]
        classA_x2 = [self.train_data[:,1][i] for i in classA_ind]
        classB_x1 = [self.train_data[:,0][i] for i in classB_ind]
        classB_x2 = [self.train_data[:,1][i] for i in classB_ind]

        # decision_boundary
        # x1 = self.train_data[:,0]
        # x2 = np.array([-(self.w[0]/self.w[1])*x - (self.w[2]/self.w[1]) for x in x1])
        x1 = self.train_data[:,0]
        for i in range(len(self.int_w)):
            part1 = self.int_w[i][0]/self.int_w[i][1]
            part2 = self.int_w[i][2]/self.int_w[i][1]
            x2 = np.array([- part1*x + part2 for x in x1])
            plt.plot(x1, x2, 'b', alpha=float(i+1)/(len(self.int_w)+1))

        plt.scatter(classA_x1, classA_x2, color='cyan', alpha=0.7)
        plt.scatter(classB_x1, classB_x2, color='purple', alpha=0.7)
        plt.ylim([-2, 2])
        plt.show()

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
