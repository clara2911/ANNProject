# import local files
from generate_data import DataBase
from ann import ANN

import numpy as np
np.random.seed(50)
# Data variables
N = 6
n = int(N/2)  # 2 because we have n*2 data
features = 2  # input vectors / patterns

mA = np.array([ 1.0, 0.5])
sigmaA = 0.2
mB = np.array([-1.0, 0.0])
sigmaB = 0.2
plot = False

# ANN parameters
verbose = False
params = {
    "learning_rate": 0.01,
    "batch_size": 6,
    "theta": 0,
    "epsilon": 0.0,  # slack for error during training
    "epochs": 100,
    "act_fun": 'step',
    "test_data": None,
    "test_targets": None,
    "m_weights": 0.1,
    "sigma_weights": 0.5,
    "nodes": 1
}


# make data
X, Y = DataBase.make_data(n, features, mA, mB, sigmaA, sigmaB, plot=plot)
# train ANN
# NOTE: ANN assumes data of same class are grouped and are inputted sequentially!
ann = ANN(X, Y, **params)
ann.train(verbose=verbose)

ann.plot_decision_boundary()
# ann.plot_error_history()
