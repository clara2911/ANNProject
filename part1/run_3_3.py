# import local files
from generate_data import DataBase
from ann import ANN
import matplotlib.pylab as plt
import numpy as np
np.random.seed(50)

database = DataBase()
patterns, targets = database.make_3D_data()

print('hola')
