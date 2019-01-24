# import local files
from generate_data import DataBase
from ann import ANN
import matplotlib.pylab as plt
import numpy as np

# np.random.seed(50)
# Data variables
N = 100  # Size of training dataset
features = 2  # input vectors / patterns

test_N = 20  # number of test data

plot_data = False
plot_dec = True  # plot decision_boundary

mA = np.array([ 1.0, 0.5])
sigmaA = 0.2
mB = np.array([-1.0, 0.0])
sigmaB = 0.2

linearly_separable = True

mA_non_lin = np.array([ 1.0, 0.3])
sigmaA_non_li = 0.2
mB_non_li = np.array([ 0.0, -0.1])
sigmaB_non_li = 0.2


# # COMPARISON BETWEEN PERCEPTRON AND DELTA RULE USING BATCH =================================
# # ANN parameters
# verbose = True
# params = {
#     "learning_rate": 0.001,
#     "batch_size": 6,
#     "theta": 0,
#     "epsilon": 0.0,  # slack for error during training
#     "epochs": 200,
#     "act_fun": 'step',
#     "test_data": None,
#     "test_targets": None,
#     "m_weights": 0.1,
#     "sigma_weights": 0.5,
#     "nodes": 1,
#     "learn_method": 'perceptron'
# }
#
# result = {}
#
# # make data
# data_base = DataBase()
# X, Y = data_base.make_data(n, features, mA, mB, sigmaA, sigmaB, plot=plot)
#
# # train ANN
# # NOTE: ANN assumes data of same class are grouped and are inputted sequentially!
#
# #perceptron
# ann_P = ANN(X, Y, **params)
# ann_P.train(verbose=verbose)
# ann_P.plot_decision_boundary_sequence(scatter = True)
#
# #delta rule
# params['learn_method'] = 'delta_rule'
# ann_D = ANN(X, Y, **params)
# ann_D.train(verbose=verbose)
# ann_D.plot_decision_boundary_sequence(scatter = True)
#
# ann_P.plot_decision_boundary(ann_list=[ann_D])

# COMPARISON BETWEEN PERCEPTRON AND DELTA RULE RESPECT LEARNING RATE =================================
# ANN parameters
verbose = True

data_base = DataBase()
X, Y = data_base.make_data(n, features, mA, mB, sigmaA, sigmaB, plot=plot)

fig, ax = plt.subplots()
eta = np.linspace(0.0005, 0.0015, 5)
for e in eta:
    params = {
        "learning_rate": e,
        "batch_size": 6,
        "theta": 0,
        "epsilon": 0.0,  # slack for error during training
        "epochs": 100,
        "act_fun": 'step',
        "test_data": None,
        "test_targets": None,
        "m_weights": 0,
        "sigma_weights": 0.5,
        "nodes": 1,
        "learn_method": 'perceptron' #'delta_rule'
    }

    ann = ANN(X, Y, **params)
    ann.train(verbose=verbose)
    ax.plot(range(len(ann.error_history)), ann.error_history, label='$\eta = {}$'.format(e))
    #ax.set_xlim(0, 40)
ax.legend()
plt.show()


# make data
if (linearly_separable):
    X, Y = DataBase.make_data(N, features, mA, mB, sigmaA, sigmaB, plot=plot_data)
else:
    X, Y = DataBase.make_non_lin_data(N, features, mA_non_lin, mB_non_li, sigmaA_non_li, sigmaB_non_li, plot=plot_data)
# train ANN
# NOTE: ANN assumes data of same class are grouped and are inputted sequentially!
ann = ANN(X, Y, **params)
ann.train(verbose=verbose, plot_dec=plot_dec)

exit()
if (linearly_separable):
    test_X, test_Y = DataBase.make_data(N, features, mA, mB, sigmaA, sigmaB, plot=plot_data)
else:
    test_X, test_Y = DataBase.make_non_lin_data(N, features, mA_non_lin, mB_non_li, sigmaA_non_li, sigmaB_non_li, plot=plot_data)
ann.test(test_X, test_Y, plot_dec=plot_dec)
