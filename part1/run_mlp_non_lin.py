# import local files
from generate_data import DataBase
# from mlp import MLP
from two_layer_non_lin import MLP
import matplotlib.pylab as plt
import numpy as np

sampleA = 1.0
sampleB = 1.0
subsamples=False

N = 200  # Max data is 200
ndata = int(N/2)  # per class

add_bias = True
plot_data = False

data_base = DataBase()
X, Y = data_base.non_linear_data(ndata, sampleA=sampleA, sampleB=sampleB, subsamples=subsamples, add_bias=add_bias, plot=plot_data)
test_X, test_Y = data_base.non_linear_data(ndata, sampleA=sampleA, sampleB=sampleB, subsamples=subsamples, add_bias=add_bias, plot=plot_data)

verbose = True
params = {
    "learning_rate": 0.01,
    "batch_size": N,  # setting it as 1 means sequential learning
    "theta": 0,
    "epsilon": 0.0,  # slack for error during training
    "epochs": 1000,
    "m_weights": 0.0,
    "sigma_weights": 0.2,
    "beta": 1
}

def compare_hidden_nodes():

    list_hidden_nodes = [2, 5, 10, 20]

    fig, ax = plt.subplots()

    error = 'mse'  # 'mse' , 'miss'

    for hidden_nodes in list_hidden_nodes:
        # train ANN
        NN_structure = {
            0: hidden_nodes,  # hidden layer
            1: 1  # output layer
        }

        params["theta"] = 0.5  # TODO: FIX THIS
        mlp = MLP(X, Y, NN_structure, **params)
        out = mlp.train(verbose=verbose)
        ax.plot(range(len(mlp.error_history[error])), mlp.error_history[error], label='hidden nodes = {}$'.format(hidden_nodes))
        #ax.set_xlim(0, 40)
    ax.legend()
    plt.show()


def train_test():

    NN_structure = {
        0: 10,  # hidden layer
        1: 1  # output layer
    }

    params["theta"] = 0.5
    mlp = MLP(X, Y, NN_structure, **params)

    mlp.train(val_data=test_X, val_targets=test_Y, verbose=verbose)

    plot_meshgrid(mlp, X)
    # mlp.test(test_X, test_Y)

def compare_batch_seq():
    params["batch_size"] = N  # setting it as 1 means sequential learning
    train_test()
    params["batch_size"] = 1  # setting it as 1 means sequential learning
    train_test()


def plot_meshgrid(mlp, test):
    xmin, xmax = np.min(test[:, 0]), np.max(test[:, 0])
    ymin, ymax = np.min(test[:, 1]), np.max(test[:, 1])
    x = np.linspace(xmin, xmax, 50)
    y = np.linspace(ymin, ymax, 50)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    plt.scatter(test[:,0], test[:,1])
    test = np.hstack((X.ravel().reshape((-1,1)), Y.ravel().reshape((-1,1))))
    test = np.hstack((test, np.ones((len(Y.ravel()),1))*-1 ))
    Z = mlp.forward_pass(test).reshape((len(x), len(x)))
    CS = ax.contour(X, Y, Z)
    plt.show()

train_test()
# compare_hidden_nodes()
