# -*- coding: utf-8 -*-
"""
Multi-Layer Perceptron usage

@author: Raymundo Cassani
April 2017
"""

size_layers = [784, 400, 10]
activation_funct = 'relu'
reg_lambda = 1

import pickle, gzip
import numpy as np
import mlp

# Load the pickled version of the  MNIST dataset
# Downloaded from: http://deeplearning.net/data/mnist/mnist.pkl.gz
with gzip.open('mnist.pkl.gz', 'rb') as f:
    # As 'mnist.pkl.gz' was created in Python2, 'latin1' encoding is needed
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# For sake of time, the MLP is trained with the valid_set (10K examples)
# testing is performed with the test_set (10K examples)

# Training data
train_mlp_X = valid_set[0]
train_mlp_y = valid_set[1]  
# Test data
test_mlp_X = test_set[0]
test_mlp_y = test_set[1] 

# Create MLP
mlp_classifier = mlp.Mlp(size_layers, activation_funct, reg_lambda)
# Train MLP
mlp_classifier.train(train_mlp_X, train_mlp_y, 100)
# Training Accuracy
y_hat = mlp_classifier.predict(train_mlp_X)
acc = np.mean(1 * (y_hat == train_mlp_y))
print('Training Accuracy: ' + str(acc*100))
# Test Accuracy
y_hat = mlp_classifier.predict(test_mlp_X)
acc = np.mean(1 * (y_hat == test_mlp_y))
print('Testing Accuracy: ' + str(acc*100))    
