"""
Just some notes on the structure of the assignment
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

# Report performance using:
# - MSE: For each image compute the MSE of original input and reconstructed input
# - Testing: sample one image from each class and plot side-by-side original and reconstructed
# Clara: Done and can be found in plot.py --> plot_loss and plot_images

# -----------------------
# ---  Clara's notes  ---
# Auto-encoder MSE < 0.05 yields recognizable digit reconstruction.
# why doesn't it overfit for undercomplete eg 0.9 ? train loss is not well below val_loss.
# I would expect it to overfit on the train_data a little bit at least?
# Why are the reconstructed images inverted?
# ----------------------

"""
Experiments for 3.1: Single-layer Autoencoder
"""

"""1: Learning rate"""
# Experiment with learning rate
# 1) Experiment with different learning rates and plot
# 2) Add learning rate decay oer epochs (parameter in optimizer)

"""2: ReLU"""
# study ReLU activations in encoder layer
# Remember: Output layer should always use sigmoid (output range is [0, 1])
# comment on how they compare with the sigmoidal units, especially
# in terms of the convergence rate (particularly interesting for the overcomplete
# autoencoders).

"""3: hidden layer size"""
# Test different number of hidden units
# h_u < input dims (undercomplete)
# h_u = input dims
# h_u > input dims (overcomplete)
# for the overcomplete case use a regularizer (L1 or L2) during training
# compare image result for different sizes: use plot_images

"""3b: Sparseness"""
# compare the average sparseness of the hidden layer for different number of hidden nodes
# assume some low threshold below which the absolute value of the unit activation is neglected
# you can use np.where(l < thresh, 0,1) for this
# average sparseness of layer = zero_units / number of units in layer
# discuss how the higher degree of sparseness can be promoted in hidden layer representations

"""4: Denoising auto-encoders"""
# ONLY for the undercomplete autoencoders employ denoising autoencoders meaning
# the target outputs are the original input samples with additive gaussian noise
# y = x + gaussian noise (test with m=0.  s=0.1-0.3)

"""5: Plot node weights"""
# Plot weights of units
# Reshape weight vector to matrix
# Plot for 100, 200, 400 hidden nodes for the hidden layer


"""
Experiments for 3.2: Classification using a stacked pre-trained autoencoder

The point of this part is to use a Deep Neural Network with the hidden layers pretrained as an Autoencoder
Then use this representation as input to a logistic regression classification model

# train using greedy layer by layer pre-training

# Add an output layer to the DNN where Output layer dims = num of classes
"""

"""1: number of layers """
# Compare the classification performance obtained with different number of hidden layers (1,2 and 3).
# Also look at classification performance of 0 layers: Just simple classification layer
# (?logistic regression) on raw input

"""2: number of nodes"""
# As the size of hidden layers, first choose the optimal
# number of nodes in the rst hidden layer based on your experiences in the previous
# task (3.1) and then decide on the size of the other layers within a similar
# range (a bit more or less; the classication performance on the training or validation
# data subset should guide this process)

""""3: effect of each layer"""
# Look at the performance after each layer
# - reconstruction error
# - classification error

""""4: representations"""
# Examine the hidden layer representations
# (beyond the first hidden layer already studied in the previous task).
# Observe the effect of images representing dierent digits on hidden units in the
# hidden layers.

"""5: compute time"""
# Do some tests for compute time aspects

"""6: Optional: convolutional"""
# test the effect of making it a convnet instead of dense layers


