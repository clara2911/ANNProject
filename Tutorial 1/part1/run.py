#!/usr/bin/env python
"""
Main file for learning with a single layer perceptron
It contains 3 tests:
- the difference between perceptron learning rule and delta learning rule
- the influence of the learning rate
- the influence of the bias

Authors: Clara Tump, Kostis SZ and Romina Azz
"""

from generate_data import DataBase
from ann import ANN
import matplotlib.pylab as plt
import numpy as np

# Data variables
N = 200
n = int(N / 2)  # 2 because we have n*2 data
test_N = 50
test_n = int(test_N / 2)

features = 2

linearly_separable = False
add_bias = True  # add bias to the inputs

plot_data = False

verbose = True

def main():
    data_base = DataBase()
    if (linearly_separable):
        mA = np.array([ 1.0, 0.5])
        sigmaA = 0.2
        mB = np.array([-1.0, 0.0])
        sigmaB = 0.2
    else:
        mA = np.array([ -0.3, 0.0])
        sigmaA = 0.4
        mB = np.array([ 0.2, 1.1])
        sigmaB = 0.4
    X, Y = data_base.make_data(n, features, mA, mB, sigmaA, sigmaB, plot=plot_data, add_bias=add_bias)
    test_X, test_Y = data_base.make_data(test_n, features, mA, mB, sigmaA, sigmaB, plot=plot_data, add_bias=add_bias)

    # choose one of the 4 experiments
    compare_perc_delta(X, Y, test_X, test_Y)
    # compare_learning_rate(X,Y)
    # bias_influence(X,Y)

     #compare_non_linear()  # Generates specific set of non linearly separable data

def compare_perc_delta(X, Y, test_X=None, test_Y=None):
    """
    COMPARISON BETWEEN PERCEPTRON AND DELTA RULE USING BATCH
    :param X: the input data (N (number of inputs) x M (number of features before bias)
    :param Y: the output targets (1 x N)
    :param X: Training inputs
    :param Y: Training targets
    :return: three plots: two showing the transition of the boundary for delta rule and perceptron
    and a third one comparing the final boundaries for both learning methods.

    The use of batch and sequential learning has to be changed manually
    """

    params = {
        "learning_rate": 0.001,
        "batch_size": N,
        "theta": 0,
        "epsilon": 0.0,  # slack for error during training
        "epochs": 100,
        "act_fun": 'step',
        "m_weights": 0.5,
        "sigma_weights": 0.5,
        "nodes": 1,
        "learn_method": 'perceptron'
    }

    training_method = 'batch'  # 'batch' , 'sequential'

    line_titles = ['Intermediate Boundaries', 'Final Boundary']
    #perceptron
    ann_P = ANN(X, Y, **params)
    ann_P.train(training_method, verbose=verbose)
    ann_P.plot_decision_boundary(scatter = True, # scatter data points: True/False
                               ann_list = None, # list of different models to compare
                               data=ann_P.train_data,
                               plot_intermediate=True, # plot boundary after every epoch True/False
                               title='Perceptron Decision Boundary', # title for plot
                               line_titles=line_titles,
                               data_coloring=None, # color data points as targets or predictions
                               origin_grid = False)
    #ann_P.test(test_X, test_Y)

    #delta rule
    params['learn_method'] = 'delta_rule'
    ann_D = ANN(X, Y, **params)
    ann_D.train(training_method, verbose=verbose)
    ann_D.plot_decision_boundary(scatter = True, # scatter data points: True/False
                               ann_list = None, # list of different models to compare
                               data=ann_D.train_data,
                               plot_intermediate=True, # plot boundary after every epoch True/False
                               title='Delta Rule Decision Boundary', # title for plot
                               line_titles=line_titles,
                               data_coloring=None, # color data points as targets or predictions
                               origin_grid = False)

    line_titles = ['Delta Rule', 'Perceptron']
    ann_P.plot_decision_boundary(scatter = True, # scatter data points: True/False
                               ann_list = [ann_D], # list of different models to compare
                               data=ann_P.train_data,
                               plot_intermediate=False, # plot boundary after every epoch True/False
                               title='Perceptron vs Delta Rule', # title for plot
                               line_titles=line_titles,
                               data_coloring=ann_P.train_targets, # color data points as targets or predictions
                               origin_grid = False)
    error_1 = ann_D.test(test_X, test_Y)
    error_2 = ann_P.test(test_X, test_Y)


def compare_learning_rate(X, Y):
    """
    This function studies the convergence while varying the learning rate
    :param X: Training inputs
    :param Y: Training targets
    :return: plot of the evolution of the error along the iterations
    for different values of the learning rate

    For this function the update rule and the method (batch/sequential) has to be changed
    manually in params "learn_method" and in the training function respectively

    """

    fig, ax = plt.subplots()
    eta = np.linspace(0.0005, 0.0015, 5)
    for e in eta:
        params = {
            "learning_rate": e,
            "batch_size": N,
            "theta": 0,
            "epsilon": -0.1,  # slack for error during training
            "epochs": 10,
            "act_fun": 'step',
            "m_weights": 0.9,
            "sigma_weights": 0.9,
            "nodes": 1,
            "learn_method": 'delta_rule' #'delta_rule'
        }

        training_method = 'sequential'  # 'batch' , 'sequential'

        ann = ANN(X, Y, **params)

        ann.train(training_method, verbose=verbose)

        ax.plot(range(len(ann.error_history)), ann.error_history, label='$\eta = {}$'.format(e))
        #ax.set_xlim(0, 40)
    ax.legend()
    plt.show()

def compare_non_linear(sampleA = 1.0, sampleB=1.0, subsamples=False):
    """
    This program generates the data through non_linear_data function
    and finds the boundary using batch and delta rule

    :param sampleA: percentage of data from class A that is going to be used. 100% = 100 samples.
    :param sampleB: percentage of data from class B that is going to be used. 100% = 100 samples.
    :param subsamples: boolean variable that indicates to the function to make a
        special sample to cover part 4 of question 3.1.3
    :return: plot of the decision boundary
    """
    ndata = 200
    data_base = DataBase()
    X, Y = data_base.non_linear_data(ndata, sampleA=sampleA, sampleB=sampleB, subsamples=subsamples, add_bias=True)
    mask = np.where(Y == 1)[0]
    mask2 = np.where(Y == -1)[0]
    data_base.plot_data(classA=X[mask,:].T, classB=X[mask2,:].T)
    params = {
        "learning_rate": 0.001,
        "batch_size": 6,
        "theta": 0,
        "epsilon": 0.0,  # slack for error during training
        "epochs": 300,
        "act_fun": 'step',
        "test_data": None,
        "test_targets": None,
        "m_weights": 0,
        "sigma_weights": 0.5,
        "nodes": 1,
        "learn_method": 'delta_rule'
    }

    training_method = 'batch'

    ann = ANN(X, Y, **params)

    ann.train(training_method, verbose=verbose)

    if subsamples:
        title = '80% from classA(1,:)<0 and 20% from classA(1,:)>0'
    else:
        title = '{}% from class A and {}% from class B'.format(sampleA*100, sampleB*100)
    ann.plot_decision_boundary_general(scatter = True, data=ann.train_data, targets=ann.train_targets, title=title)

    #compute sensitivity and specificity
    targets = ann.train_targets
    estimations = ann.predictions
    TP = 0.0
    FN = 0.0
    TN = 0.0
    FP = 0.0
    for i in range(len(ann.predictions)):
        if targets[i] == 1 and estimations[i] == 1:
            TP += 1.0
        elif targets[i] == 1 and estimations[i] == -1:
            FN += 1.0
        elif targets[i] == -1 and estimations[i] == 1:
            FP += 1.0
        else:
            TN += 1.0
    TPR = TP / (FN + TP)
    TNR = TN / (FP + TN)
    print('TPR: {} and TNR: {}'.format(TPR, TNR))

def bias_influence(X,Y):
    '''
    illustrate the capabilities of the model without the addition of bias
    Using batch learning and the delta_rule
    :param X: the input data (N (number of inputs) x M (number of features before bias)
    :param Y: the output targets (1 x N)
    '''
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

    training_method = 'sequential'  # 'batch' , 'sequential'

    ann = ANN(X, Y, **params)

    ann.train(training_method, verbose=verbose)

    if (params["bias"] == 0):
        title = 'Learning without bias'
    else:
        title = 'Learning with bias'

    ann.plot_decision_boundary(
            data=ann.train_data,
            plot_intermediate=True,
            title=title,
            data_coloring = ann.train_targets,
            origin_grid=True
            )

if __name__ == "__main__":
    main()
