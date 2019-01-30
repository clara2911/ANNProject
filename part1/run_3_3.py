# import local files
from generate_data import DataBase
from ann import ANN
import matplotlib.pylab as plt
import numpy as np
from two_layer import MLP

np.random.seed(50)

database = DataBase()
patterns, targets = database.make_3D_data(bias=True, plot_data=False)
N = 21


# Currently for some reason it works for very specific parameter values
params = {
    "learning_rate": 0.001,
    "batch_size": N,  # setting it as 1 means sequential learning
    "epsilon": 0.0,  # slack for error during training
    "epochs": 5000,
    "act_fun": 'sigmoid',
    "m_weights": 0.0,
    "sigma_weights": 0.2,
    "beta": 1
}

verbose = 1

NN_structure = {
    0: 25,  # hidden layer
    1: 1  # output layer
}

mlp = MLP(patterns, targets, NN_structure, **params)
targets_pred = mlp.train(verbose=verbose, validation=False, plot_error=True, plot_at_500=True) #, activation_function = mlp.linear

# no need to do another forward_pass. we already have targets_pred
# train_out = mlp.forward_pass(patterns)
n = int(np.sqrt(len(targets_pred)))
x = np.arange(-5, 5.5, 0.5)
y = np.arange(-5, 5.5, 0.5)
X, Y = np.meshgrid(x, y)
Z = targets_pred.reshape((n,n))

#plot the objective function
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')
plt.show()
