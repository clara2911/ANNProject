# import local files
from generate_data import DataBase
from mlp import MLP
import matplotlib.pylab as plt
import numpy as np


# Data variables
N = 4
n = int(N/2)  # 2 because we have n*2 data
features = 2  # input vectors / patterns

mA = np.array([ 1.0, 0.5])
sigmaA = 0.2
mB = np.array([-1.0, 0.0])
sigmaB = 0.2
plot_data = False

test_n = 50
test_mA = np.array([ 1.0, 0.5])
test_sigmaA = 0.2
test_mB = np.array([-1.0, 0.0])
test_sigmaB = 0.2

data_base = DataBase()
X, Y = data_base.make_data(n, features, mA, mB, sigmaA, sigmaB, plot=plot_data)
test_X, test_Y = data_base.make_data(test_n, features, test_mA, test_mB, test_sigmaA, test_sigmaB, plot=plot_data)

#in here we use batch
verbose = True
params = {
    "learning_rate": 0.001,
    "batch_size": 6,
    "theta": 0,
    "epsilon": 0.0,  # slack for error during training
    "epochs": 1000,
    "act_fun": 'step',
    "test_data": None,
    "test_targets": None,
    "m_weights": 0.1,
    "sigma_weights": 0.2,
    "nodes": 10,
    "learn_method": 'perceptron'
}

# train ANN
# NOTE: ANN assumes data of same class are grouped and are inputted sequentially!

#perceptron
mlp_P = MLP(X, Y, **params)
mlp_P.train_batch(verbose=verbose)
#mlp_P.train_sequential(verbose=verbose)
# mlp_P.plot_decision_boundary_sequence(scatter = True, data=mlp_P.train_data, targets=mlp_P.train_targets)
mlp_P.test(test_X, test_Y)

#delta rule
params['learn_method'] = 'delta_rule'
mlp_D = MLP(X, Y, **params)
mlp_D.train_batch(verbose=verbose)
# mlp_D.train_sequential(verbose=verbose)
#mlp_D.plot_decision_boundary_sequence(scatter = True, data=mlp_D.train_data, targets=mlp_D.train_targets)

#mlp_P.plot_decision_boundary(ann_list=[mlp_D], data=mlp_P.train_data, targets=mlp_P.train_targets)

error_1 = mlp_D.test(test_X, test_Y)
error_2 = mlp_P.test(test_X, test_Y)
