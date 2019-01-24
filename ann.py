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
            "learn_method": 'perceptron',
            "error": 1.0,
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
        X = np.hstack((X, bias_vec))  #  changed so bias is after (before it was beginning)
        return X, Y


    def train(self, verbose=False, plot_dec=False):
        """
        Train neural network
        """
        self.train_data_size = self.train_data.shape[0]
        for iteration in range(self.epochs):
            #initialize batch indices
            start_batch = (iteration * self.batch_size) % self.train_data_size
            end_batch = (start_batch + self.batch_size) % self.train_data_size  # WATCHOUT: this might become smaller than start batch
            #take a random batch or sequential batch?
            data = self.train_data[start_batch:end_batch]
            targets = self.train_targets[start_batch:end_batch]
            #predict (compute product of X * w)
            self.sum = np.dot(data, self.w)  # Y_pred: NxNeurons  "the output will be an NxNeurons matrix of sum for each neuron for each input vector"

            # compute delta_w
            delta_w = self.learn(data, targets)

            if self.error <= self.epsilon:
                if (verbose):
                    self.print_info(iteration, self.error)
                break
                
            #update
            self.w = self.w - delta_w
            self.int_w.append(self.w)

            # maybe even compute MSE?
            if (verbose):
                self.print_info(iteration, self.error)
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

    def learn(self, data, targets):
        function = self.learning_method[self.learn_method]
        return function(data, targets)

    def perceptron(self, data, targets):
        """
        Perceptron learning
        """
        #pass result through activation function
        self.predictions = self.activation_function(targets)
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


    def step(self, targets):
        """
        Set predictions to 1 or -1 depending on threshold theta
        """
        Y_threshed = np.where(self.sum > self.theta, 1, -1)  #TODO why not targets?
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

    def plot_decision_boundary_sequence(self, scatter = True):

        fig, ax = plt.subplots()
        classA_ind = np.where(self.predictions > 0)[0]
        classB_ind = np.where(self.predictions <= 0)[0]
      
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

    def plot_decision_boundary(self, data, targets, weights, scatter=True, ann_list=None):
        """
        Plot data as classified from the NN and the sequence
        of decision boundaries of the weights
        """
        fig, ax = plt.subplots()
        classA_ind = np.where(targets > 0)[0]
        classB_ind = np.where(targets< 0)[0]

        classA_x1 = [data[:,0][i] for i in classA_ind]
        classA_x2 = [data[:,1][i] for i in classA_ind]
        classB_x1 = [data[:,0][i] for i in classB_ind]
        classB_x2 = [data[:,1][i] for i in classB_ind]
        
        
        # decision_boundary
        x1 = data[:,0]
        #x1 = np.arange(np.min(self.train_data[:, 0]), np.max(self.train_data[:, 0]), 0.1)
        #         for i in range(weights.shape[1]):  weights[0][i]
        part1 = weights[0] / weights[1]
        part2 = weights[2] / weights[1]
        x2 = np.array([- part1 * x + part2 for x in x1])
        # x2 = np.array([- part1*x - self.bias* part2 for x in x1])
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
