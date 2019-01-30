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
        self.learning_methods = {
            'perceptron': self.perceptron,
            'delta_rule': self.delta_rule
        }
        self.error_functions = {
            'perceptron': self.missclass_error,
            'delta_rule': self.mse_error
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
            "learning_method": 'perceptron',
            "error": 1.0,
            "bias": -1
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.n_features = data.shape[1]  # Dimensionality of data / Features / Attributes / Input vectors
        self.train_data, self.train_targets = data, targets
        self.w = self.init_weights()
        self.error_history = []


    def init_weights(self):
        """
        Initialize weight matrix Features x Neurons
        """
        w = np.random.normal(self.m_weights, self.sigma_weights, self.n_features)

        if (self.nodes == 1):
            w = w.reshape(-1,1)
        else:
            for j in range(self.nodes - 1):  # -1 because you have already set one dimension
                w_j = np.random.normal(self.m_weights, self.sigma_weights, self.n_features)
                w = np.vstack((w, w_j))
            w = w.T

        return w


    def train_batch(self, verbose=False):
        """
        Train neural network
        """
        iteration = 0
        self.int_w = []
        self.error_history = []

        data = self.train_data
        targets = self.train_targets
        for iteration in range(self.epochs):
            #predict (compute product of X * w)
            sum = np.dot(data, self.w)  # Y_pred: NxNeurons "the output will be an NxNeurons matrix of sum for each neuron for each input vector"

            # compute delta_w
            delta_w, error = self.learn(data, sum, targets)

            self.int_w.append(self.w)  # iteration 0 has random weights
            self.error_history.append(error)

            if (verbose):
                self.print_info(iteration, error)
            if error <= self.epsilon:
                break

            #update
            self.w = self.w - delta_w

            # maybe even compute MSE?
        self.int_w = np.vstack(self.int_w)


    def train_sequential(self, verbose=False):
        """
        Train neural network
        """
        data = self.train_data
        targets = self.train_targets

        self.int_w = [self.w]
        self.error_history = []
        self.predictions = np.empty(len(targets))

        for iteration in range(self.epochs):

            for nd, d in enumerate(data):
                #predict (compute product of X * w)
                sum = np.dot(d, self.w)  # Y_pred: NxNeurons  "the output will be an NxNeurons matrix of sum for each neuron for each input vector"
                # compute delta_w
                delta_w, _ = self.learn(d, sum, targets[nd], num_data = nd)
                #update
                self.w = self.w - delta_w.reshape(-1,1)

            # update epoch weight
            self.int_w.append(self.w)
            # compute training error
            error = self.test(data, targets)
            self.error_history.append(error)

            if (verbose):
                self.print_info(iteration, error)
            if error <= self.epsilon:
                break

        self.int_w = np.vstack(self.int_w)


    def test(self, test_data, test_targets, plot_dec=False):
        """
        Test trained ANN
        """
        test_predictions = np.dot(test_data, self.w)
        targets_pred = self.activation_function(test_predictions)

        error = self.compute_error(targets_pred, test_targets)

        if (plot_dec):
            self.plot_decision_boundary(data=test_data, targets=targets_pred)

        return error


    def learn(self, data, sum, targets, num_data=None):
        """
        Use one of the learning methods to update weights
        """
        function = self.learning_methods[self.learning_method]
        X = np.transpose(data)
        return function(X, sum, targets, num_data=num_data)


    def perceptron(self, data, sum, targets, num_data=None):
        """
        Perceptron learning
        """
        error = 0.0
        #pass result through activation function
        if num_data==None:
            self.predictions = self.activation_function(sum)
            diff = self.predictions - targets
            delta_w = self.learning_rate * np.dot(data, diff)
            # compute error
            error = self.missclass_error(self.predictions, targets)
        else:
            self.predictions[num_data] = self.activation_function(sum)
            diff = self.predictions[num_data] - targets
            delta_w = self.learning_rate * np.multiply(data , diff)

        return delta_w, error


    def delta_rule(self, data, sum, targets, num_data=None):
        """
        Delta rule for computing delta w
        """

        error = 0.0
        if num_data == None:
            self.predictions = self.activation_function(sum)
            diff = sum - targets
            delta_w = self.learning_rate * np.dot(data, diff)
            # compute error
            error = np.mean(diff ** 2)  # mse
        else:
            self.predictions[num_data] = self.activation_function(sum)
            diff = sum - targets
            delta_w = self.learning_rate * np.multiply(data, diff)

        return delta_w, error


    def step(self, sum):
        """
        Set predictions to 1 or -1 depending on threshold theta
        """
        Y_threshed = np.where(sum > self.theta, 1, -1)
        return Y_threshed


    def activation_function(self, sum):
        """
        Pass predictions through an activation function
        """
        function = self.activation_functions[self.act_fun]
        return function(sum)


    def compute_error(self, predictions, targets):
        """
        Compute the error dependine on the learning method
        perceptron - missclassification
        delta_rule - MSE
        """
        function = self.error_functions[self.learning_method]
        return function(predictions, targets)


    def mse_error(self, predictions, targets):
        """
        Calculate Mean Squared Error
        """
        diff = predictions - targets
        error = np.mean(diff ** 2)
        return error


    def missclass_error(self, predictions, targets):
        """
        Calculate percentage of missclassification
        """
        miss = len(np.where(predictions != targets)[0])
        return float(miss/len(targets))


    def plot_decision_boundary_sequence(self, scatter = True, data=None, targets=None):
        """
        Plot data as classified from the NN and the sequence
        of decision boundaries of the weights
        """
        fig, ax = plt.subplots()
        classA_ind = np.where(self.predictions > 0)[0]
        classB_ind = np.where(self.predictions <= 0)[0]

        classA_x1 = [data[:,0][i] for i in classA_ind]
        classA_x2 = [data[:,1][i] for i in classA_ind]
        classB_x1 = [data[:,0][i] for i in classB_ind]
        classB_x2 = [data[:,1][i] for i in classB_ind]

        # decision_boundary
        x1 = data[:, 0]
        for i in range(self.int_w.shape[1]):
            part1 = self.int_w[0][i] / self.int_w[1][i]
            part2 = self.int_w[2][i] / self.int_w[1][i]
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


    def plot_decision_boundary(self, scatter = True, ann_list = None, data=None, targets=None):
        """
        Plot data as classified from the NN and the decision boundary of the weights
        """
        fig, ax = plt.subplots()
        if targets is not None:
            classA_ind = np.where(targets > 0)[0]
            classB_ind = np.where(targets <= 0)[0]
        else:
            classA_ind = np.where(self.predictions > 0)[0]
            classB_ind = np.where(self.predictions <= 0)[0]

        classA_x1 = [data[:,0][i] for i in classA_ind]
        classA_x2 = [data[:,1][i] for i in classA_ind]
        classB_x1 = [data[:,0][i] for i in classB_ind]
        classB_x2 = [data[:,1][i] for i in classB_ind]

        # decision_boundary
        x1 = data[:, 0]
        part1 = self.w[0] / self.w[1]
        part2 = self.w[2] / self.w[1]
        x2 = np.array([- part1 * x + part2 for x in x1])
        ax.plot(x1, x2, '--', alpha=0.5, label = self.learning_method)

        if ann_list:
            for ann in ann_list:
                part1 = ann.w[0] / ann.w[1]
                part2 = ann.w[2] / ann.w[1]
                x2 = np.array([- part1 * x - part2 for x in x1])
                ax.plot(x1, x2, '--', alpha=0.5, label=ann.learning_method)

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
