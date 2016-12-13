# -*- coding: utf-8 -*-
"""  MSE functions  """


import numpy as np


# Function calculating the Mean Square Error (MSE) of an error vector e.
# 
#     e is the error vector
#

"""
# Returns
# 
#     MSE(e) (the mean square error of e)
# 
"""

def calculate_mse (e):
    
    return np.mean (e.T.dot (e)) / 2


# Function computing the loss of a prediction function w on data set y and x, using MSE.
# 
#     y is the output
#     tx is the input
#
#     w is the prediction function
#

"""
# Returns
# 
#     L(w) (the MSE loss of prediction function w)
# 
"""

def compute_mse_loss (y, tx, w):
    
    e = y - tx.dot (w)
    return calculate_mse (e)


# Function computing the gradient of the loss function of a prediction function w.
# 
#     y is the output
#     tx is the input
#
#     w is the prediction function
#

"""
# Returns
# 
#     grad (the gradient of the loss function (MSE) for y, tx and w)
# 
"""

def compute_mse_gradient (y, tx, w):
    
    e = y - tx.dot (w)
    return -tx.T.dot (e) / y.shape [0]