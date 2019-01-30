# import local files
from generate_data import DataBase
from ann import ANN
import matplotlib.pylab as plt
import numpy as np
from two_layer import MLP

np.random.seed(50)

database = DataBase()
patterns, targets = database.make_3D_data()
N = 21

params = {
    "learning_rate": 0.001,
    "batch_size": N,  # setting it as 1 means sequential learning
    "theta": 0,
    "epsilon": 0.0,  # slack for error during training
    "epochs": 7000,
    "act_fun": 'sigmoid',
    "m_weights": 0.1,
    "sigma_weights": 0.2,
    "beta": 1
}

verbose = 1

NN_structure = {
    0: 20,  # hidden layer
    1: 1  # output layer
}

params["theta"] = 0.  # step threshold
mlp = MLP(patterns, targets, NN_structure, **params)
targets_pred = mlp.train(verbose=verbose, validation=False) #, activation_function = mlp.linear

plt.plot(range(len(mlp.error_history['mse'])), mlp.error_history['mse'])
plt.show()

train_out = mlp.forward_pass(patterns)
n = int(np.sqrt(len(targets_pred)))
Z = train_out.reshape((n,n))
x = np.arange(-5, 5.5, 0.5)
y = np.arange(-5, 5.5, 0.5)
X, Y = np.meshgrid(x, y)

#plot the objective function
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')
plt.show()

print('hola')





print('hola')
