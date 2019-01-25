import collections
import numpy as np
from matplotlib import pyplot as plt

class MLP:

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
            "beta": 1.0,
            "bias": -1
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.n_features = data.shape[1]


        self.train_data, self.train_targets = self.shape_input(data, targets)

        layers = collections.OrderedDict()
        layers['0'] = self.n_features + 1
        layers['1'] = 5
        layers['2'] = 3
        layers['3'] = self.train_data.shape[0]

        self.num_of_hidden_layers = 3
        self.weights = self.init_weights(layers)
        self.error_history = []


    def init_weights(self, layers):
        """
        Initialize weight matrix Features x Neurons
        """
        layers_list = []

        for layer, weights_per_layer in layers.items():
            w_l = np.random.normal(self.m_weights, self.sigma_weights, weights_per_layer)
            layers_list.append(w_l)
        return layers_list

    def shape_input(self, X, Y):
        """
        Add bias as input vector (feature) in the data and shuffle them
        """
        index_shuffle = np.random.permutation(X.shape[0])
        X = X[index_shuffle]
        Y = Y[index_shuffle]
        bias_vec = self.bias * np.ones((X.shape[0], 1))  # put a minus in front
        X = np.hstack((X, bias_vec))  #  changed so bias is after (before it was beginning)
        return X, Y

    def train_batch(self, verbose=False):
        """
        Train neural network
        """
        self.error_history = []

        self.alpha_layer_out = [None] * self.num_of_hidden_layers  # Output layer is NOT considered a HIDDEN LAYER

        for iteration in range(self.epochs):

            data = self.train_data  # Data dimensions: Features + 1 x N_batch_size
            targets = self.train_targets  # N_batch_size

            # FORWARD PASS STARTS HERE
            for feature in range(self.train_data.shape[1]):  # for every input vector / feature
                # for every input vector / feature
                data = self.train_data[:,feature]

                input_of_layer = data.reshape(-1,1)  # Shape it so that N_batch_size x 1

                # INPUT LAYER AND HIDDEN LAYERS
                for layer in range(self.num_of_hidden_layers):  # layer index depicted as zeta

                    # WEIGHTS ARE MATRIX (Different column for every feature?) OR JUST ONE COLUMN PER LAYER?
                    w_zeta = self.weights[layer].reshape(-1,1).T  # Shape it so that 1 x L/M (Number of nodes (Neurons) of layer)

                    # data: N_batch_size x 1
                    # weights (input layer): 1 x L/M nodes (NEURONS) of layer
                    h_zeta_matrix = np.dot(input_of_layer, w_zeta)  # h_zeta: N_batch_size x L/M  (eq 4.4)

                    h_zeta = np.sum(h_zeta_matrix, axis=0)  # L/M x 1  (eq 4.4)

                    h_zeta = h_zeta.reshape(-1, 1)  # 1 x L/M

                    alpha_zeta = self.sigmoid_function(h_zeta, beta=self.beta)  # output of layer zeta

                    input_of_layer = alpha_zeta  # the output of the current layer is the input of the next

                    self.alpha_layer_out[layer] = alpha_zeta  # thresholded output of zeta layer
                    self.weights[layer] = w_zeta  # weights of zeta layer

                # OUTPUT LAYER
                w_kapa = self.weights[-1].reshape(-1,1).T  # Layer k weights (Hidden layer): 1 x M_k current layer's hidden nodes (NEURONS)

                h_kapa_matrix = np.dot(input_of_layer, w_kapa)  # (1 x M_(k-1)) * (M_k x 1) = M_(k-1) x (M_k)

                h_kapa = np.sum(h_kapa_matrix, axis=0)  # M_k x 1

                h_kapa = h_kapa.reshape(-1, 1)  # 1 x M_k

                alpha_kapa = self.sigmoid_function(h_kapa)  # output of layer k

                self.weights[-1] = w_kapa  # weights of k layer

                outputs = alpha_kapa

                # BACKPROP STARTS HERE

                # compute Output ERROR
                delta_out_k = (outputs - targets) * outputs * (1.0 - outputs)  # delta_out: N_batch_size x 1

                # compute Hidden error
                delta_h = [None] * self.num_of_hidden_layers

                # first -1: we ignore the output layer, second -1: we want to include 0 index, third -1: iterate from last to first
                for layer in range(self.num_of_hidden_layers-1, -1, -1):  # layer index depicted as zeta

                    part1 = self.alpha_layer_out[layer] * (1.0 - self.alpha_layer_out[layer])  # 1 x M_(zeta)

                    w_zeta = self.weights[layer]  # 1 x M_zeta

                    # is it np.dot(delta_out) or np.dot(previous_delta) ? in the book its delta_out but this is for 2 layers!
                    # ^ it IS delta_out! checked here https://ecee.colorado.edu/~ecen4831/lectures/NNet3.html
                    part2_matrix = np.dot(delta_out_k, w_zeta)  # (1 x N_batch_size) * (1 x M_zeta) = N_batch_size x M_zeta

                    part2 = np.sum(part2_matrix, axis=0)  # M_zeta x 1

                    part2 = part2.reshape(-1, 1)  # 1 x M_zeta

                    delta_h_zeta = part1 * part2  # 1 x M_zeta * 1 x M_zeta = 1 x M_zeta

                    delta_h[layer] = delta_h_zeta


                # update output layer
                input_of_output_layer = self.alpha_layer_out[-1]  # output of layer before output = input of output layer

                for node in range(len(self.weights[-1])):
                    update_out_w = 0
                    for input in input_of_output_layer:
                        update_out_w += self.learning_rate * input * delta_out_k[node]
                    self.weights[-1][node] = self.weights[-1][node] - update_out_w

                # TODO: WATCHOUT: Check if this is correct
                # update_out_w = self.learning_rate * np.dot(input_of_output_layer, delta_out_k)  # (1 x M_prev_layer) * (1 x M_k) = 1 x M_k (output nodes)
                # self.weights[-1] = self.weights[-1] - update_out_w  # TODO: Is it - or +? page 78: - , page 79: +


                # update hidden layers
                # first -1: we ignore the output layer, 0: we DONT want to include 0 index, third -1: iterate from last to first
                for layer in range(self.num_of_hidden_layers-1, 0, -1):  # layer index depicted as zeta
                    input_of_layer = self.alpha_layer_out[layer - 1]  # output of previous layer is input of current layer

                    for node in range(len(self.weights[layer])):
                        update_w_k = 0
                        for input in input_of_layer:
                            update_w_k += self.learning_rate * input * delta_h[layer][node]
                        self.weights[layer][node] = self.weights[layer][node] - update_w_k

                    # update_h_w = self.learning_rate * np.dot(input_of_layer, delta_h[hidden_layer])
                    # self.weights[hidden_layer] = self.weights[hidden_layer] - update_h_w  # Is it - or +? page 78: - , page 79: +

                # update first layer
                # update_in_w = self.learning_rate * np.dot(data, delta_h[0])
                # self.weights[0] = self.weights[0] - update_in_w
                for node in range(len(self.weights[0])):
                    update_w_k = 0
                    for input in data:
                        update_w_k += self.learning_rate * input * delta_h[0][node]
                    self.weights[0][node] = self.weights[0][node] - update_w_k



    def test(self, test_data, test_targets):
        """
        Test trained ANN
        """
        test_data, test_targets = self.shape_input(test_data, test_targets)
        test_predictions = np.dot(test_data, self.weights)
        self.sum = test_predictions
        targets_pred = self.activation_function()

        error = self.missclass_error(targets_pred, test_targets)

        print('Test Error: ', error)
        return error


    def learn(self, data, targets, num_data=None):
        function = self.learning_method[self.learn_method]
        return function(data, targets, num_data=num_data)

    def sigmoid_function(self, h_zeta, beta=1.0):
        """
        Compute the sigmoid function given a threshold beta
        """
        denominator = 1 + np.exp(-beta * h_zeta)
        return 1 / denominator

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
        for i in range(len(self.int_w)):  # for every
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

    def plot_decision_boundary(self, scatter = True, ann_list = None, data=None, targets=None):
        """
        Plot data as classified from the NN and the decision boundary of the weights
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
