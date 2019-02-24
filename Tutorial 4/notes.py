"""
Just some notes on the structure of the assignment
"""

"""
Load data from local cvs
"""

"""
Setup DNN
"""

# Weight matrix 28x28 with small normally distributed random values (m=0.  s=0.1)
# biases of all layers initialized to 0

# Use as an activation function the sigmoid or ReLU for hidden layers
# Output layer should always use sigmoid (output range is [0, 1])

# Use one hidden layer in 3.1
# I started with 32 hidden nodes

# Output layer dims = Input Layer dims for 3.1
# Output layer dims = num of classes for 3.2

# Train using gradient descent loss
# Clara: all of the above is done and can be found in autoencoder.py

"""
Experiments for 3.1: Shallow Autoencoder
"""

"""1"""
# Experiment with learning rate
# 1) Experiment with different learning rates and plot
# 2) Add learning rate decay oer epochs (parameter in optimizer)

"""0"""
# study ReLU activations in encoder layer
# Remember: Output layer should always use sigmoid (output range is [0, 1])
# comment on how they compare with the sigmoidal units, especially
# in terms of the convergence rate (particularly interesting for the overcomplete
# autoencoders).

"""2"""
# Test different number of hidden units
# h_u < input dims (undercomplete)
# h_u = input dims
# h_u > input dims (overcomplete)
# for the overcomplete case use a regularizer (L1 or L2) during training

"""3"""
# Test how ReLU and sigmoid compare in the hidden layers in terms of convergence rate
# (especially for overcomplete autoencoders)

"""4"""
# ONLY for the undercomplete autoencoders employ denoising autoencoders meaning
# the target outputs are the original input samples with additive gaussian noise
# y = x + gaussian noise (test with m=0.  s=0.1-0.3)


"""5"""
# For each image compute the MSE of original input and reconstructed input
# Save each MSE and at the end of epoch get the mean and std to calculate the epoch's MSE
# At end of training plot each epoch's mean and std MSE

"""6"""
# Testing: sample one image from each class and plot side-by-side original and reconstructed

"""7"""
# Try different sizes of the hidden layer representation and compare errors.

"""8"""
# Get a trained NN and for each layer measure the average sparseness.
# For each layer:  (this can be done with np.where(units < 0.1, 0, 1)
#   For each unit:
#       if unit is below threshold (~0.1?)
#           dead_units += 1
#   average sparseness of layer = dead_units / number of units in layer
# "discuss how the higher degree of sparseness can be promoted in hidden layer representations"

"""9"""
# Plot weights of units
# Reshape weight vector to matrix
# Plot for 100, 200, 400 hidden nodes


"""
Experiments for 3.2: Classification using a deep pre-trained autoencoder

The point of this part is to use a Deep Neural Network with the hidden layers pretrained as an Autoencoder
Then use this representation to the final output layer for classification
"""

"""1"""
# Add an output layer to the DNN where Output layer dims = num of classes

"""2"""
# Compare the classification performance obtained with different number of hidden layers (1,2 and 3).

"""3"""
