# import local files
from generate_data import DataBase
from ann import ANN
import matplotlib.pylab as plt
import numpy as np
np.random.seed(50)
# Data variables
N = 200
n = int(N/2)  # 2 because we have n*2 data
features = 2  # input vectors / patterns

mA = np.array([ 1.0, 0.5])
sigmaA = 0.2
mB = np.array([-1.0, 0.0])
sigmaB = 0.2
plot = False

data_base = DataBase()
X, Y = data_base.make_data(n, features, mA, mB, sigmaA, sigmaB, plot=plot)

# COMPARISON BETWEEN PERCEPTRON AND DELTA RULE USING BATCH =================================
# ANN parameters
def compare_perc_delta(X, Y):
    """

    :param X: Training inputs
    :param Y: Training targets
    :return: three plots: two showing the transition of the boundary for delta rule and perceptron
    and a third one comparing the final boundaries for both learning methods.

    The use of batch and sequential learning has to be changed manually
    """
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
        "sigma_weights": 0.5,
        "nodes": 1,
        "learn_method": 'perceptron'
    }

    # train ANN
    # NOTE: ANN assumes data of same class are grouped and are inputted sequentially!

    #perceptron
    ann_P = ANN(X, Y, **params)
    #ann_P.train_batch(verbose=verbose)
    ann_P.train_sequential(verbose=verbose)
    ann_P.plot_decision_boundary_sequence(scatter = True, data=ann_P.train_data, targets=ann_P.train_targets)

    #delta rule
    params['learn_method'] = 'delta_rule'
    ann_D = ANN(X, Y, **params)
    #ann_D.train_batch(verbose=verbose)
    ann_D.train_sequential(verbose=verbose)
    ann_D.plot_decision_boundary_sequence(scatter = True, data=ann_D.train_data, targets=ann_D.train_targets)

    ann_P.plot_decision_boundary(ann_list=[ann_D], data=ann_P.train_data, targets=ann_P.train_targets)


# COMPARISON BETWEEN PERCEPTRON AND DELTA RULE RESPECT LEARNING RATE =================================
# ANN parameters
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
    X, Y = data_base.non_linear_data(sampleA=sampleA, sampleB=sampleB, subsamples=subsamples)
    mask = np.where(Y == 1)[0]
    mask2 = np.where(Y == -1)[0]
    data_base.plot_data(classA=X[mask,:].T, classB=X[mask2,:].T)
    params = {
        "learning_rate": 0.001,
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
        "learn_method": 'delta_rule'
    }

    ann = ANN(X, Y, **params)
    ann.train_batch(verbose=True)
    if subsamples:
        title = '20% from classA(1,:)<0 and 80% from classA(1,:)>0'
    else:
        title = '{}% from class A and {}% from class B'.format(sampleA*100, sampleB*100)
    ann.plot_decision_boundary(scatter = True, data=ann.train_data, targets=ann.train_targets, title=title)


#compare_learning_rate(X,Y)
#compare_perc_delta(X, Y)
compare_non_linear(sampleA = 1.0, sampleB=1.0, subsamples=True)
