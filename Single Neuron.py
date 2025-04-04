#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 08:56:27 2025

@author: jaredsills
"""

import numpy as np

def sigmoid(x):
    # Our activation function aka sigmoid function
    # f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# create the number of weights, in this case, two weights for two inputs
weights  = np.array([0,1])
bias = 4

# create the neuron
n = Neuron(weights, bias)

# creat a vector of two inputs
x = np.array([2,3])
print(n.feedforward(x))
