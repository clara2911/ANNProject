import collections
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error as sk_mse

class MLP:

    def __init__(self, data, targets, structure, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """
        var_defaults = {
            "learning_rate": 0.01,
            "batch_size": 1,
            "theta": 0,
            "epsilon": 0.0,
            "epochs": 100,
            "m_weights": 0.1,
            "sigma_weights": 0.05,
            "nodes": 1,
            "beta": 1.0,
            "bias": -1
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.n_features = data.shape[1]

        self.train_data, self.train_targets = data, targets

        self.num_of_hidden_layers = 1
        self.weights = self.init_weights(structure)
        self.error_history = {'mse': [], 'miss': []}


    def init_weights(self, structure):
        """
        Initialize weight matrix Features x Neurons
        For a two layer network
        """
        layers_list = [None] * (self.num_of_hidden_layers + 1)  # +1 is the output layer

        first_layer_weights = []
        for node in range(structure[0]):
            first_layer_weights.append(np.random.normal(self.m_weights, self.sigma_weights, self.n_features))
        layers_list[0] = np.array(first_layer_weights)

        w_l = []
        dim_out_prev_layer = structure[0]
        for node in range(structure[1]):
            w_l.append(np.random.normal(self.m_weights, self.sigma_weights, dim_out_prev_layer))
        layers_list[1] = np.array(w_l)
        # layers_list[1] = np.array(layers_list[layer])
        return np.array(layers_list)

    def train(self, verbose=False, validation=True, plot_error=False, plot_at_500=False, use_best_weights=False):
        """
        Train neural network
        """

        self.validation_error_during_train = {'mse':[], 'miss':[]}

        self.alpha_layer_out = [None] * self.num_of_hidden_layers  # Output layer is NOT considered a HIDDEN LAYER

        self.min_train_error = 1.0
        for iteration in range(self.epochs):

            for i in range(0, self.train_data.shape[0], self.batch_size):  # for every input vector

                start = i
                end = i + self.batch_size

                data = self.train_data[start:end]  # Data dimensions: Features + 1 x N_batch_size
                targets = self.train_targets[start:end]  # N_batch_size

                out = self.forward_pass(data)
                self.backward_pass(data, targets, out)

            train_out = self.forward_pass(self.train_data)

            mse_error, miss_error = self.compute_error(train_out, self.train_targets)
            self.error_history['mse'].append(mse_error)
            self.error_history['miss'].append(miss_error)

            if (mse_error < self.min_train_error):
                self.best_weights = self.weights
                self.min_train_error = mse_error

            if validation:
                val_out = self.forward_pass(self.validation_data)
                val_mse_error, val_miss_error = self.compute_error(val_out, self.val_targets)
                self.validation_error_during_train['mse'].append(val_mse_error)
                self.validation_error_during_train['miss'].append(val_miss_error)

            if (plot_at_500 and iteration%500==0):

                n = int(np.sqrt(len(train_out)))
                x = np.arange(-5, 5.5, 0.5)
                y = np.arange(-5, 5.5, 0.5)
                X, Y = np.meshgrid(x, y)
                Z = train_out.reshape((n,n))

                #plot the objective function
                ax = plt.axes(projection='3d')
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                cmap='viridis', edgecolor='none')
                ax.set_title('surface')
                plt.show()

                if (plot_error):
                    self.plot_error_history(self.error_history)

        if (use_best_weights):
            self.weights = self.best_weights
            train_out = self.forward_pass(self.train_data)

            mse_error, miss_error = self.compute_error(train_out, self.train_targets)
            self.error_history['mse'].append(mse_error)
            self.error_history['miss'].append(miss_error)

        if (plot_error):
            self.plot_error_history(self.error_history)
            if validation:
                self.plot_error_history(self.validation_error_during_train)
        return train_out

    def compute_error(self, out, targets):
        mse_error = self.mse(out, targets)
        out_thres = self.step(out)
        miss_error = self.missclass_error(out_thres, targets)
        return mse_error, miss_error

    def forward_pass(self, data):
        """
        Calculate outputs on weights
        """

        # WEIGHTS ARE MATRIX (Different column for every feature?) OR JUST ONE COLUMN PER LAYER?
        input_of_layer = data
        w_hidden = np.array(self.weights[0]).T # Shape it so that Features x L/M (Number of nodes (Neurons) of layer)

        # data: N_batch_size x D + 1
        # weights (input layer): D + 1 x L/M nodes (NEURONS) of layer
        h_in = np.dot(input_of_layer, w_hidden)  # h_zeta: N_batch_size x L/M  (eq 4.4)
        self.h_in = h_in

        h_out = self.sigmoid_function(h_in, beta=self.beta)  # output of layer zeta

        self.alpha_layer_out[0] = h_out  # thresholded output of zeta layer

        # OUTPUT LAYER
        w_kapa = np.array(self.weights[-1]).T  # Layer k weights (Hidden layer): 1 x M_k current layer's hidden nodes (NEURONS)

        o_in = np.dot(h_out, w_kapa)  # (N x L) * (L x M)
        self.o_in = o_in


        out = self.sigmoid_function(o_in, beta=self.beta)

        return out


    def backward_pass(self, data, targets, out):
        """
        Calculate the error for all the weights and update them using generalized Delta rule
        """
        # compute Output ERROR
        delta_out_k = (out - targets) * out * (1.0 - out)  # delta_out: N_batch_size x M
        delta_out_k = delta_out_k.T

        # compute Hidden error
        w_kapa = self.weights[-1].T  # output layer's weights
        h_out = self.alpha_layer_out[0]  #

        part1 = h_out * (1.0 - h_out)  # N x L
        part2 = np.dot(w_kapa, delta_out_k).T
        delta_h = part1 * part2

        # UPDATING !!!
        # update output layer
        update_out_w = self.learning_rate * np.dot(h_out.T, delta_out_k.T) #[node])

        # update first layer
        update_w_k = self.learning_rate * np.dot(data.T, delta_h)

        self.weights[-1] = np.array(self.weights[-1]) - update_out_w.T #+ 0.01*np.array(self.weights[-1])
        self.weights[0] = np.array(self.weights[0]) - update_w_k.T #+ 0.01*np.array(self.weights[0])


    def backward_pass_ignore(self, data, targets, out):
        """
        Backward pass impleneted following the assignments pseudocode
        Currently not in use
        """
        part1_o = out - targets

        #part2_o = ( (1 + self.o_in) * (1 - self.o_in) ) * 0.5
        part2_o = ( (1 + out) * (1 - out) ) * 0.5
        delta_o = part1_o * part2_o  # np.dot(part1_o, part2_o)

        v = self.weights[-1].T  # is it v? or something else?
        delta_o = delta_o.T
        h_out = self.alpha_layer_out[0]
        part1_h = np.dot(v, delta_o)  # v * delta_o
        #part2_h = ( (1 + self.h_in) * (1 - self.h_in) ) * 0.5
        part2_h = ( (1 + h_out) * (1 - h_out) ) * 0.5
        part2_h = part2_h.T
        delta_h = part1_h * part2_h  # np.dot(part1_h, part2_h)

        # delta_h # remove bias line?

        momentum = 0.9
        part1_dw = np.dot(delta_h, data)  # delta_h * data
        part2_dw = (1 - momentum)
        dw = - part1_dw * part2_dw # np.dot(part1_dw, part2_dw)
        dw = dw * self.learning_rate

        part1_dv = np.dot(delta_o, h_out)  # delta_o * h_out
        part2_dv = (1 - momentum)
        dv = - part1_dv * part2_dv  # np.dot(part1_dv, part2_dv)
        dv = dv * self.learning_rate

        self.weights[0] = self.weights[0] + dw
        self.weights[-1] = self.weights[-1] + dv

    def test(self, test_data, test_targets):
        """
        Test trained ANN
        """
        targets_pred = self.forward_pass(test_data)
        targets_pred_thres = self.step(targets_pred)
        error = self.missclass_error(targets_pred_thres, test_targets)

        print('Test Error: ', error)
        return targets_pred_thres, error


    def sigmoid_function(self, h_zeta, beta=1.0):
        """
        Compute the sigmoid function given a threshold beta
        1. / denominator >>> out:[0, 1]  (theta should be 0.5)
        ( 2. / denominator) - 1 >>> out: [-1, 1]  (theta should be 0.)
        (( 2. / denominator) - 1 ) * 0.5 >>> out: [-0.5, 0.5]  (theta should be 0.)
        """
        denominator = 1 + np.exp(-beta * h_zeta)
        return 1. / denominator

    def step(self, y):
        """
        Set predictions to 1 or -1 depending on threshold theta
        """
        Y_threshed = np.where(y > self.theta, 1, -1)
        return Y_threshed

    def linear(self, y):
        """
        Linear threshold, f(x) = x (no transformation)
        """
        return y

    def missclass_error(self, predictions, targets):
        """
        Calculate percentage of missclassification
        """
        miss = len(np.where(predictions != targets)[0])
        return float(float(miss)/len(targets))

    def encoder_error(self, predictions, targets):
        """
        Calculate percentage of missclassification in an encoder
        """
        correct_count = 0
        total_count = 0
        for i_r, row_t in enumerate(targets):
            for i_e, elem_t in enumerate(row_t):
                total_count += 1
                if (predictions[i_r][i_e] == targets[i_r][i_e]):
                    correct_count += 1

        return correct_count, total_count

    def mse(self, predictions, targets):
        """
        Calculate Mean Squered Error
        """
        mse = sk_mse(predictions, targets)
        return mse

    def plot_error_history(self, error_history, plot_miss=False):
        """
        Plot the history of the error (show how quickly the NN converges)
        """
        x_axis_mse = range(1, len(error_history['mse']) + 1)
        y_axis_mse = error_history['mse']

        fig, ax = plt.subplots()

        custom_lines = [Line2D([0], [0], color='purple'), Line2D([0], [0], color='red')]
        ax.legend(custom_lines, ['missclassification', 'MSE'], frameon=False, loc='upper right')
        ax.set_title('Error plots')

        if (plot_miss):
            x_axis_miss = range(1, len(error_history['miss']) + 1)
            y_axis_miss = error_history['miss']
            plt.plot(x_axis_miss, y_axis_miss, color='purple', alpha=0.7)

        plt.plot(x_axis_mse, y_axis_mse, color='red', alpha=0.7)
        plt.show()

    def print_info(self, iteration, error):
        print('Iter: {}'.format(iteration))
        print('Train Error: {}'.format(error))
        print('Weights: {}'.format(self.w))
        print('\n')
