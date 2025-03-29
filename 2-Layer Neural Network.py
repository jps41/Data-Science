#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 07:12:27 2025

@author: jaredsills
"""

import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/ (1+np.exp(-x))

# input dataset
X = np.array([ [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]])

# output dataset
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just good practice)
np.random.seed(1)

# initialize weights randomly ith mean 0
syn0 = 2*np.random.random((3,1)) - 1

for itr in range(1000):
    
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    
    # how much did we miss?
    l1_error = y - l1
    
    # multiply how much we missed by the slope
    # of the sigmoid function at the values in layer 1
    l1_delta = l1_error * nonlin(l1,True)
    
    # update weights
    syn0 += np.dot(l0.T, l1_delta)
    
    # print every 200th iteration to visualize weight change
    if (itr% 200) == 0:
        #print(f'{l0=}')
        #print(f'{l1=}')
        #print(f'{l1_error=}')
        print(f'{l1_delta=}')
    
print("Output After Training")
print(l1)

    


