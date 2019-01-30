# import local files
from generate_data import DataBase
# from mlp import MLP
from two_layer_non_lin import MLP
import matplotlib.pylab as plt
import numpy as np

sampleA = 1.0
sampleB=1.0
subsamples=False

N = 200  # Max data is 200
ndata = int(N/2)  # per class

add_bias = True
plot_data = True

data_base = DataBase()
#X, Y = data_base.non_linear_data(ndata, sampleA=sampleA, sampleB=sampleB, subsamples=subsamples, add_bias=add_bias, plot=plot_data)
#test_X, test_Y = data_base.non_linear_data(ndata, sampleA=sampleA, sampleB=sampleB, subsamples=subsamples, add_bias=add_bias, plot=plot_data)

mA = np.array([ 1.0, 0.5])
sigmaA = 0.2
mB = np.array([-1.0, 0.0])
sigmaB = 0.2

features = 2

X, Y = data_base.make_data(ndata, features, mA, mB, sigmaA, sigmaB, plot=plot_data, add_bias=add_bias)
test_X, test_Y = data_base.make_data(ndata, features, mA, mB, sigmaA, sigmaB, plot=plot_data, add_bias=add_bias)

verbose = True
params = {
    "learning_rate": 0.01,
    "batch_size": N,  # setting it as 1 means sequential learning
    "theta": 0,
    "epsilon": 0.0,  # slack for error during training
    "epochs": 10000,
    "m_weights": 0.0,
    "sigma_weights": 0.1,
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

    params["theta"] = 0.  # step threshold
    mlp = MLP(X, Y, NN_structure, **params)
    mlp.train(val_data=test_X, val_targets=test_Y, verbose=verbose)

    mlp.test(test_X, test_Y)

def compare_batch_seq():
    params["batch_size"] = N  # setting it as 1 means sequential learning

    params["batch_size"] = 1  # setting it as 1 means sequential learning

train_test()
# compare_hidden_nodes()
