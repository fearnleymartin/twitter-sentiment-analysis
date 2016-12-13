# -*- coding: utf-8 -*-
"""  MAE functions  """


import numpy as np


# Function calculating the Mean Absolute Error (MAE) of an error vector e.
# 
#     e is the error vector
#

"""
# Returns
# 
#     MAE(e) (the mean absolute error of e)
# 
"""

def calculate_mae (e):
    
    return np.mean (np.abs (e))


# Function computing the loss of a prediction function w on data set y and x, using MAE.
# 
#     y is the output
#     tx is the input
#
#     w is the prediction function
#

"""
# Returns
# 
#     L(w) (the MAE loss of prediction function w)
# 
"""

def compute_mae_loss (y, tx, w):
    
    e = y - tx.dot (w)
    return calculate_mae (e)


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
#     grad (the gradient of the loss function (MAE) for y, tx and w)
# 
"""

def compute_mae_gradient (y, tx, w):
    
    e = y - tx.dot (w)
    grad = np.zeros (len (w))
    coeff = 1 / len (y)
    
    for i in range (len (w)):
        
        gi = 0
        
        for j in range (len (y)):
            
            if e [j] > 0:
                gi -= tx [j, i]
            elif e [j] < 0:
                gi += tx [j, i]
                
        gi *= coeff
        grad [i] = gi
    
    return grad