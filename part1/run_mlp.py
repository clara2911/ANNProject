# import local files
from generate_data import DataBase
# from mlp import MLP
from two_layer import MLP
import matplotlib.pylab as plt
import numpy as np


# Data variables
N = 100
n = int(N / 2)  # 2 because we have n*2 data
test_N = 20
test_n = int(test_N / 2)
features = 2  # input vectors / patterns

# Linearly separable data
mA = np.array([ 1.0, 0.5])
sigmaA = 0.2
mB = np.array([-1.0, 0.0])
sigmaB = 0.2
plot_data = False

# Non linearly separable data
non_mA = [1.0, 0.3]
non_sigmaA = 0.2
non_mB = [0.0, -0.1]
non_sigmaB = 0.3

data_base = DataBase()
X, Y = data_base.make_data(n, features, mA, mB, sigmaA, sigmaB, plot=plot_data)
test_X, test_Y = data_base.make_data(test_n, features, mA, mB, sigmaA, sigmaB, plot=plot_data)

#X, Y = data_base.non_linear_data(n, non_mA, non_mB, non_sigmaA, non_sigmaB)
#test_X, test_Y = data_base.non_linear_data(n, non_mA, non_mB, non_sigmaA, non_sigmaB)

#in here we use batch
verbose = True
params = {
    "learning_rate": 0.01,
    "batch_size": 10,  # setting it as 1 means sequential learning
    "theta": 0,
    "epsilon": 0.0,  # slack for error during training
    "epochs": 1000,
    "act_fun": 'step',
    "test_data": None,
    "test_targets": None,
    "m_weights": 0.1,
    "sigma_weights": 0.1,
    "nodes": 10,
}

# train ANN
# NOTE: ANN assumes data of same class are grouped and are inputted sequentially!

NN_structure = {
    0: 10,  # hidden layer
    1: 1  # output layer
}
mlp = MLP(X, Y, NN_structure, **params)
mlp.train(verbose=verbose)
mlp.test(test_X, test_Y)
