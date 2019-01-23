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
            "learn_meth": 'delta_rule',
            "bias": -1
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))
        self.n_features = data.shape[1]
        self.train_data, self.train_targets = self.shape_input(data, targets)
        self.w = self.init_weights()


    def init_weights(self):
        """
        Initialize weight matrix Features x Neurons
        """
        w = np.random.normal(self.m_weights, self.sigma_weights,self.n_features + 1)  # + 1 is bias

        for j in range(self.nodes - 1):
            w_j = np.random.normal(self.m_weights, self.sigma_weights, self.n_features + 1)  # + 1 is bias
            w = np.vstack((w, w_j))
        w = w.reshape(-1,1)
        return w


    def shape_input(self, X, Y):
        """
        Shuffle the data
        Add bias as an extra feature on the bottom of the input vector
        """
        index_shuffle = np.random.permutation(X.shape[0])
        X = X[index_shuffle]
        Y = Y[index_shuffle]
        bias_vec = self.bias * np.ones((X.shape[0], 1))  # put a minus in front
        X = np.hstack((X, bias_vec))  # changed so bias is after (before it was beginning)
        return X, Y


    def train(self, verbose=False, plot_dec=False):
        """
        Train neural network
        """
        iteration = 0
        error = 1.
        self.int_w = []
        self.error_history = []
        self.train_data_size = self.train_data.shape[0]
        while(error > self.epsilon and iteration < self.epochs):
            #initialize batch indices
            start_batch = (iteration * self.batch_size) % self.train_data_size
            end_batch = (start_batch + self.batch_size) % self.train_data_size  # WATCHOUT: this might become smaller than start batch
            #take a random batch or sequential batch?
            data = self.train_data[start_batch:end_batch]
            targets = self.train_targets[start_batch:end_batch]
            #predict (compute product of X * w)
            self.predictions = np.dot(data, self.w)  # Y_pred: NxNeurons  "the output will be an NxNeurons matrix of sum for each neuron for each input vector"
            #compute delta_w using the perceptron learning
            delta_w = self.learn(data, targets)
            # delta_w = self.delta_rule()
            #update
            self.w = self.w - delta_w
            #compute error
            error = self.missclass_error(self.predictions, targets)
            self.error_history.append(error)
            # maybe even compute MSE?
            self.int_w.append(self.w)
            iteration += 1
            if (verbose):
                self.print_info(iteration, error)
        self.int_w = np.vstack(self.int_w)
        if (plot_dec):
            self.plot_decision_boundary(self.train_data, self.predictions, self.int_w)


    def test(self, test_data, test_targets, plot_dec=False):
        """
        Test trained ANN
        """
        test_data, test_targets = self.shape_input(test_data, test_targets)
        test_predictions = np.dot(test_data, self.w)
        targets_pred = self.activation_function(test_predictions)

        error = self.missclass_error(targets_pred, test_targets)
        if (plot_dec):
            print('Test Error: ', error)
            self.plot_decision_boundary(test_data, targets_pred, self.w)
    def learn(self, data, targets):
        """
        Pass predictions through an activation function
        """
        function = self.learning_methods[self.learn_meth]
        return function(data, targets)

    def perceptron(self, data, targets):
        """
        Perceptron learning
        """
        #pass result through activation function
        self.predictions = self.activation_function(targets)
        delta_w = self.delta_rule(data, targets)
        return delta_w


    def delta_rule(self, data, targets):
        """
        Delta rule for computing delta w
        """
        print("self.predictions")
        print(self.predictions)
        diff = self.predictions - targets
        X = np.transpose(data)
        delta_w = self.learning_rate * np.dot(X, diff)
        return delta_w


    def step(self, targets):
        """
        Set predictions to 1 or -1 depending on threshold theta
        """
        Y_threshed = np.where(targets > self.theta, 1, -1)
        return Y_threshed


    def activation_function(self, targets):
        """
        Pass predictions through an activation function
        """
        function = self.activation_functions[self.act_fun]
        return function(targets)


    def missclass_error(self, predictions, targets):
        """
        Calculate percentage of missclassification
        """
        miss = len(np.where(predictions != targets)[0])
        return float(miss/len(targets))


    def plot_decision_boundary(self, data, targets, weights):
        """
        Plot data as classified from the NN and the decision boundary of the weights
        """

        classA_ind = np.where(targets > 0)[0]
        classB_ind = np.where(targets< 0)[0]

        classA_x1 = [data[:,0][i] for i in classA_ind]
        classA_x2 = [data[:,1][i] for i in classA_ind]
        classB_x1 = [data[:,0][i] for i in classB_ind]
        classB_x2 = [data[:,1][i] for i in classB_ind]

        # decision_boundary
        # x1 = self.train_data[:,0]
        # x2 = np.array([-(self.w[0]/self.w[1])*x - (self.w[2]/self.w[1]) for x in x1])
        x1 = data[:,0]
        for i in range(weights.shape[1]):
            part1 = weights[0][i]/weights[1][i]
            part2 = weights[2][i]/weights[1][i]
            x2 = np.array([- part1*x - self.bias* part2 for x in x1])
            plt.plot(x1, x2, 'b', alpha=float(i+1)/(len(weights)+1))

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
