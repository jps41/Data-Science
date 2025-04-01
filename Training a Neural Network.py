#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 09:46:21 2025

@author: jaredsills
"""

import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # mean square error = 1/n * Sum of (Know output - Predicited Output)^2
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork():
    '''
    Our neural network contains:
        - Two inputs (weight of person, height of person)
        - One hidden layer with two neurons (h1, h2)
        - An output layer with one neuron (o1)
    '''
    
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
    
        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        
        return o1
    
    def train(self, data, all_y_trues):
        
        '''
        - data is a (n x 2) array, n = # of samples in the dataset
        - all_y_trues is an array with n elements which correspond
            the data set
            
        '''
        learn_rate = 0.1 
        epochs = 1000 # number of training iterations 
        
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Do a feedforward 
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                # Calculae the partial derivatives
                # d_L_d_w1 = partial deriv L / partial deriv w1
                
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                
                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                
                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                
                # --- Update weights and biases
                
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -+ learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                
                # Neuorn h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -+ learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                                                 
                # Neuron 01
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -+ learn_rate * d_L_d_ypred * d_ypred_d_b3
        
                # Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_pred = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_pred)
                    print('Epoch %d loss: %.3f' % (epoch, loss))
                    
# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])             

# Train the neural network
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
jared = np.array([3,5])
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 -
print("Jared: %.3f" % network.feedforward(jared))

                

      
      
      