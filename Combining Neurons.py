#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 09:09:52 2025

@author: jaredsills
"""
import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
class OurNeuralNetwork:
    '''
    A neurol network the contains:
        - 2 inputs
        - a hidden layer with two neurons (h1, h2)
        - an output layer with one neuron (o1)
    Each neuron has the same weights and bias:
        - w =  [0,1]
        - b = 0
    '''
    def __init__(self):
        weights = np.array([0,1])
        bias = 0
        
        # Utilize the Neuron class to create the hidden layers (neurons)
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
        
    # Utilize the feedfoward function from the Neuron class
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        
        # The inputs for o1 are the outputs 
        # from h1 and h2 (out_h1 and out_h2)
        
        out_o1 = self.o1.feedforward([out_h1, out_h2])
        
        return out_o1

# test it out

network = OurNeuralNetwork()
x = np.array([2,3])
print(network.feedforward(x))

        
    
    
        


