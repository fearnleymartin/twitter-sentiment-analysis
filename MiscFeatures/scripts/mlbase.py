# -*- coding: utf-8 -*-
"""  Implementation of the basis for Machine Learning practice.

   It contains all the useful functions, as seen in the Labs, to compute
   a prediction function, to be used on test data set.
   
   To use them, you need to import either mse.py, mae.py or any other python script containing
   a custom loss function (with its function to compute the gradient), and pass those functions
   as argument.  """

import numpy as np
import scripts.helpers as hlp
import scripts.mse as mse
import scripts.loglikelihood as lkh

# Function using gradient descent to compute the weight
#
#     y is the train output
#     tx is the train input
#     
#     gamma is the step size to increment w with the gradient (smaller = more preicision but slower convergence)
#     max_iters is the number of iterations we'll do to find the local minimum
#     
#     initial_w is the start weight to go from
#
#     loss_function is the loss function to use to compute the loss
#     gradient_function is the gradient function to use to compute the gradient
#

"""
# Returns
# 
#     losses (sequence of losses for each weight computed)
#     ws (sequence of computed weights)
# 
"""


def least_squares_GD (y, tx, gamma, max_iters, initial_w = None, loss_function = mse.compute_mse_loss, gradient_function = mse.compute_mse_gradient):
    
    # Default initial w
    if initial_w == None:
        initial_w = np.zeros (tx.shape [1])
    
    # Initialize w
    w = initial_w
    
    # Define parameters to store w and loss
    ws = []
    losses = []
    
    # Iterate max_iters times to get closer to the local minimum
    for n_iter in range(max_iters):
        
        # Store current weight
        ws.append (np.copy (w))
        
        # Compute and store current loss
        loss = loss_function (y, tx, w)
        losses.append(loss)
            
        if n_iter % 100 == 0:
            print("Current iteration={i}, the loss={l}".format(i=n_iter, l=loss))
        
        # Compute gradient for the next iteration
        if n_iter < max_iters - 1:
            gw = gradient_function (y, tx, w)
            w = w - gamma * gw

    return losses, ws


# Function using stochastic gradient descent to compute the weight
#
#     y is the train output
#     tx is the train input
#     
#     gamma is the step size to increment w with the gradient (smaller = more preicision but slower convergence)
#     max_iters is the number of iterations we'll do to find the local minimum
#     
#     initial_w is the start weight to go from
#     batch_size is the size of the batch to generate for internal use
#
#     loss_function is the loss function to use to compute the loss
#     gradient_function is the gradient function to use to compute the gradient
# 

"""
# Returns
# 
#     losses (sequence of losses for each weight computed)
#     ws (sequence of computed weights)
# 
"""

def least_squares_SGD (y, tx, gamma, max_iters, initial_w = None, batch_size = None, loss_function = mse.compute_mse_loss, gradient_function = mse.compute_mse_gradient):
    
    # Default initial w
    if initial_w == None:
        initial_w = np.zeros (tx.shape [1])
        
    # Default batch_size
    if (batch_size == None):
        batch_size = len (y) / 10
    
    # Initialize w
    w = initial_w
    
    # Define parameters to store w and loss
    ws = []
    losses = []
    
    # Iterate max_iters times to get closer to the local minimum
    for minibatch_y, minibatch_tx in hlp.batch_iter (y, tx, batch_size, max_iters):
        
        # Store current weight
        ws.append (np.copy (w))
        
        # Compute and store current loss
        loss = loss_function (minibatch_y, minibatch_tx, w)
        losses.append(loss)
        
        # Compute gradient for the next iteration
        if n_iter < max_iters - 1:
            gw = gradient_function (minibatch_y, minibatch_tx, w)
            w = w - gamma * gw
        
    return losses, ws


# Function using Least Squares method to compute the weight (MSE only)
#
#     y is the train output
#     tx is the train input
# 

"""
# Returns
# 
#     w (the exact weight minimizing the loss function for given y and tx)
#     loss (the loss of this optimal w)
# 
"""

def least_squares (y, tx):
    
    w = np.linalg.inv (tx.T.dot (tx)).dot (tx.T).dot (y)
    loss = mse.compute_mse_loss (y, tx, w)
    
    return loss, w


# Function using Ridge Regression method to compute the weight (MSE only)
#
#     y is the train output
#     tx is the train input
#     
#     lambda_ is the regularization parameter
# 

"""
# Returns
# 
#     w (the exact weight minimizing the loss function for given y, tx and lambda)
#     loss (the loss of this optimal w)
# 
"""

def ridge_regression (y, tx, lambda_ = 0):
    
    w = np.linalg.inv (tx.T.dot (tx) + lambda_ * 2 * tx.shape [0] * np.eye(tx.shape [1])).dot (tx.T).dot (y)
    loss = mse.compute_mse_loss (y, tx, w)
    
    return loss, w


# Function using Logistic Regression method to compute the loss, gradient and hessian of the Log-Likelyhood cost function
#
#     y is the train output
#     tx is the train input
#     w is the weight
#     
#     lambda_ is the regularization parameter
# 

"""
# Returns
# 
#     loss (the loss of this w)
#     grad (the gradient of the loss)
# 
"""

def logistic_regression (y, tx, w):
    
    return lkh.calculate_loglikelihood_loss (y, tx, w), lkh.calculate_loglikelihood_gradient (y, tx, w), lkh.calculate_loglikelihood_hessian (y, tx, w)

# Function using Penalized Logistic Regression method to compute the loss, gradient and hessian of the cost function (Loglikelihood only)
#
#     y is the train output
#     tx is the train input
#     w is the weight
#     
#     lambda_ is the regularization parameter
# 

"""
# Returns
# 
#     loss (the loss of this w)
#     grad (the gradient of the loss)
#     hess (the hessian matrix of the loss)
# 
"""

def reg_logistic_regression (y, tx, w, lambda_):
    
    loss, grad, hess = logistic_regression (y, tx, w)
    
    loss += lambda_ * w.T.dot (w) [0] [0]
    
    lamb = 2 * lambda_
    
    grad = np.add (grad, np.multiply (w, lamb))
        
    for i in range (len (hess)):
        hess [i, i] += lamb
    
    return loss, grad, hess

# Function computing one step of the gradient descent using Loglikelihood cost function and only its gradient
#
#     y is the train output
#     tx is the train input
#     w is the weight
#     
#     alpha is the step coefficient
#     lambda_ is the regularization parameter
# 

"""
# Returns
# 
#     loss (the loss of this w)
#     w (the new weight, affected by the calculated gradient)
# 
"""

def logistic_GD_step (y, tx, w, alpha, lambda_ = 0):
    
    loss, grad, hess = reg_logistic_regression (y, tx, w, lambda_)
    w = np.subtract (w, np.multiply (grad, alpha))
    
    return loss, w

# Function computing one step of the gradient descent using Loglikelihood cost function, its gradient and its hessian
#
#     y is the train output
#     tx is the train input
#     w is the weight
#     
#     alpha is the step coefficient
#     lambda_ is the regularization parameter
# 

"""
# Returns
# 
#     loss (the loss of this w)
#     w (the new weight, affected by the calculated gradient and hessian)
# 
"""

def newton_GD_step (y, tx, w, alpha, lambda_ = 0):
    
    loss, grad, hess = reg_logistic_regression (y, tx, w, lambda_)
    
    inv = np.linalg.inv (hess).dot (grad)
    w = np.subtract (w, np.multiply (inv, alpha))
    
    return loss, w

# Function using gradient descent with penalized logistic regression to find the global minimum
# of the loglikelihood loss function, and its weight.
#
#     y is the train output
#     tx is the train input
#     
#     step_function is the step function to use for gradient descent
#     
#     max_iter is the maximum allowed iterations for the process
#     threshold is the threshold of difference between the loss of two consecutive w to consider to have found the minimum
#     
#     alpha is the step coefficient
#     lambda_ is the regularization parameter
#     
#     info_step is the interval of iterations to print out the current loss (0 means no prints)
# 

"""
# Returns
# 
#     w (the best weight, minimizing the loss function)
# 
"""

def logistic_GD (y, tx, step_function, max_iter = 10000, threshold = 1e-4, alpha = 0.01, lambda_ = 0, info_step = 100):
    
    w = np.zeros ((tx.shape[1], 1))
    lastLoss = float ('inf')

    for iter in range (max_iter):
        
        loss, w = step_function (y, tx, w, alpha, lambda_)
            
        if info_step > 0 and iter % info_step == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
            
        if np.abs (loss - lastLoss) < threshold:
            break
            
        lastLoss = loss
    
    return w

# Function using stochastic gradient descent with penalized logistic regression to find the global minimum
# of the loglikelihood loss function, and its weight.
#
#     y is the train output
#     tx is the train input
#     
#     step_function is the step function to use for gradient descent
#     
#     max_iter is the maximum allowed iterations for the process
#     max_search_error is the maximum iterations since last found minimum to consider that we won't find better
#     batch_size is the size of the batch from the input to compute for lighter gradient descent
#     threshold is the threshold of difference between the loss of two consecutive w to consider to have found the minimum
#     
#     alpha is the step coefficient
#     lambda_ is the regularization parameter
#     
#     info_step is the interval of iterations to print out the current loss (0 means no prints)
# 

"""
# Returns
# 
#     w (the best weight, minimizing the loss function)
# 
"""

def logistic_SGD (y, tx, step_function, max_iter = 10000, max_search_error = 200, batch_size = 1000, threshold = 1e-4, alpha = 0.01, lambda_ = 0.01, info_step = 100):
    
    w = np.zeros ((tx.shape[1], 1))
    lastLoss = float ('inf')
    
    minw = w
    minLoss = lastLoss
    lastMin = 0
    
    n_iter = 0
    while n_iter < max_iter:

        broken = False
        for minibatch_y, minibatch_tx in hlp.batch_iter (y, tx, batch_size):

            loss, w = step_function (minibatch_y, minibatch_tx, w, alpha, lambda_)
            
            if loss < minLoss:
                minLoss = loss
                minw = w
                lastMin = n_iter

            n_iter += 1
            
            if info_step > 0 and n_iter % info_step == 0:
                print("Current iteration={i}, the loss={l}".format(i=n_iter, l=loss))

            if np.abs (loss - lastLoss) < threshold:
                broken = True
                break
                
            if n_iter - lastMin > max_search_error:
                broken = True
                break

            lastLoss = loss
            
        if broken == True:
            break
            
    return minw