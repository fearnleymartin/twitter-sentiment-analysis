# -*- coding: utf-8 -*-
"""  Some helper functions  """

import numpy as np


# Function normalizing a given data set.
#
#     x is the data set to standardize
#     
#     mean_x is the value we want to subtract to x, default is the mean of x
#     std_x is the value we want x to be standardized with, default is the standard of x
# 

"""
# Returns
# 
#     tx (normalized data, with offset if needed)
#     
#     mean_x (as supplied, or mean of input x if none was given)
#     std_x (as supplied, or standard of input x if none was given)
#
"""

def standardize (x, forget_first_column = True):
    
    if forget_first_column:
        
        x_mean = np.mean (x [:, 1:], axis = 0)
        x_std = np.std (x [:, 1:], axis = 0)
        
        return np.c_ [x [:, 0], ((x [:, 1:] - x_mean) / x_std)]
    
    else:
    
        x_mean = np.mean (x, axis = 0)
        x_std = np.std (x, axis = 0)

        return ((x - x_mean) / x_std)


# Function generating a minibatch iterator for a dataset.
# 
#     y is the output
#     tx is the input
#     
#     batch_size is the size of the minibatch we want
#     num_batches is the number of batches we want to perform calculus on
#
#     shuffle should be True if we want to pick random values instead of whole blocks of values
# 

"""
# Returns
# 
#     An iterator over minibatches of our sample y and tx
#
"""

def batch_iter (y, tx, batch_size, num_batches = None, shuffle = True):
    
    # Compute a default batch count
    data_size = len (y)
    num_batches_max = int (np.ceil(data_size/batch_size))
    
    # Define the true batch count (lower the one supplied by the user if too big)
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min (num_batches, num_batches_max)

    # Shuffle the input and output if needed
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
        
    # Create the iterator
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# Function building a polynomial matrix based on input x
#
#     x is the input
#     degree is the max degree of polynomial values
#     

"""
# Returns
# 
#     phi (A matrix of polynomial elements from x)
#
"""

""" 
# Example : given [[x11, x12], [x21, x22], [x31, x32]] as x and 2 as degree, the output will be:
#      _                      _
#     | 1  x11  x11² x12  x12² |
#     | 1  x21  x21² x22  x22² |
#     |_1  x31  x31² x32  x32²_|
#
"""
            
def build_poly (x, degree):
    
    # First column = only 1s
    phi = np.array ([[1 for i in range (x.shape [0])]]).T
    
    if degree == 0:
        return phi
    
    # Iterate through each dimension of x
    for j in range (x.shape [1]):
        
        # Iterate through each power
        for k in range (degree):
            
            if k == 0:
                phi = np.c_ [phi, x [:, j]]
                continue
                
            phi = np.c_ [phi, np.multiply (phi [:, j * degree + k], x [:, j])]
        
    return phi
