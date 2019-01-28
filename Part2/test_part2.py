import h5py
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(5,)))
# model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))

filename = 'best_model.h5'
f = model.load_weights(filename)
print(f)
