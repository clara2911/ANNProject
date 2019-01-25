# import local files
from generate_data import DataBase
from ann import ANN
import matplotlib.pylab as plt
import numpy as np


def main():
    # Data variables
    N = 200
    n = int(N/2)  # 2 because we have n*2 data
    features = 2  # input vectors / patterns

    mA = np.array([ 1.0, 0.5])
    sigmaA = 0.2
    mB = np.array([-1.0, 0.0])
    sigmaB = 0.2

    test_n = 50
    test_mA = np.array([ 1.0, 0.5])
    test_sigmaA = 0.2
    test_mB = np.array([-1.0, 0.0])
    test_sigmaB = 0.2
    plot_data = False

    data_base = DataBase()
    X, Y = data_base.make_data(n, features, mA, mB, sigmaA, sigmaB, plot=plot_data)
    test_X, test_Y = data_base.make_data(test_n, features, test_mA, test_mB, test_sigmaA, test_sigmaB, plot=plot_data)

    # compare_perc_delta(X, Y)
    #compare_learning_rate(X,Y)
    bias_influence(X,Y)

# COMPARISON BETWEEN PERCEPTRON AND DELTA RULE USING BATCH =================================
# ANN parameters
def compare_perc_delta(X, Y):
    #in here we use batch
    verbose = True
    params = {
        "learning_rate": 0.001,
        "batch_size": 6,
        "theta": 0,
        "epsilon": 0.0,  # slack for error during training
        "epochs": 200,
        "act_fun": 'step',
        "test_data": None,
        "test_targets": None,
        "m_weights": 0.1,
        "sigma_weights": 0.2,
        "nodes": 2,
        "learn_method": 'perceptron'
    }

    # train ANN
    # NOTE: ANN assumes data of same class are grouped and are inputted sequentially!

    #perceptron
    ann_P = ANN(X, Y, **params)
    ann_P.train_batch(verbose=verbose)
    #ann_P.train_sequential(verbose=verbose)
    ann_P.plot_decision_boundary_sequence(scatter = True, data=ann_P.train_data, targets=ann_P.train_targets)
    ann_P.test(test_X, test_Y)

    #delta rule
    params['learn_method'] = 'delta_rule'
    ann_D = ANN(X, Y, **params)
    ann_D.train_batch(verbose=verbose)
    # ann_D.train_sequential(verbose=verbose)
    ann_D.plot_decision_boundary_sequence(scatter = True, data=ann_D.train_data, targets=ann_D.train_targets)

    ann_P.plot_decision_boundary(ann_list=[ann_D], data=ann_P.train_data)

    error_1 = ann_D.test(test_X, test_Y)
    error_2 = ann_P.test(test_X, test_Y)

# COMPARISON BETWEEN PERCEPTRON AND DELTA RULE RESPECT LEARNING RATE =================================
# ANN parameters
def compare_learning_rate(X, Y):
    # here we use batch
    verbose = True

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
        #ann.train_batch(verbose=verbose)
        ann.train_sequential(verbose=verbose)
        ax.plot(range(len(ann.error_history)), ann.error_history, label='$\eta = {}$'.format(e))
        #ax.set_xlim(0, 40)
    ax.legend()
    plt.show()

# illustrate the capabilities of the model without the addition of bias
def bias_influence(X,Y):
    verbose = False
    params = {
        "learning_rate": 0.1,
        "batch_size": 1,
        "theta": 0,
        "epsilon": 0.0,  # slack for error during training
        "epochs": 50,
        "act_fun": 'step',
        "test_data": None,
        "test_targets": None,
        "m_weights": 0,
        "sigma_weights": 0.5,
        "nodes": 1,
        "learn_method": 'delta_rule',
        "bias": 0
    }
    ann = ANN(X, Y, **params)
    ann.train_batch(verbose=verbose)
    ann.plot_decision_boundary(
            data=ann.train_data,
            plot_intermediate=True,
            title='Learning without bias',
            data_coloring = ann.train_targets,
            origin_grid=True
            )

if __name__ == "__main__":
    main()

