import numpy as np


# Function sigmoid
#

"""
# Returns
# 
#     the sigmoid of input t
# 
"""

def sigmoid (t):
    
    if t > 15:
        return 1
    elif t < -15:
        return 0
    
    e = np.exp (t)
    return e / (1 + e)


# Function calculating the loglikelihood loss of a weight vector w
#

"""
# Returns
# 
#     loss (the loglikelihood loss of the supplied w vector)
# 
"""

def calculate_loglikelihood_loss (y, tx, w):
    
    lw = 0
    
    for i in range (len (y)):
        
        v = tx [i].T.dot (w) [0]
    
        if v > 15:
            lw += (1 - y [i]) * v
        elif v < -15:
            lw -= y [i] * v
        else:
            lw += np.log (1 + np.exp (v)) - y [i] * v
        
    return lw


# Function calculating the loglikelihood gradient on a weight vector w
#

"""
# Returns
# 
#     grad (the loglikelihood gradient on the supplied w vector)
# 
"""

def calculate_loglikelihood_gradient (y, tx, w):
    
    s = tx.dot (w)
    
    for i in range (len (s)):
        s [i] = sigmoid (s [i]) - y [i]
        
    return tx.T.dot (s)


# Function calculating the loglikelihood hessian matrix on a weight vector w
#

"""
# Returns
# 
#     hess (the loglikelihood hessian matrix on the supplied w vector)
# 
"""

def calculate_loglikelihood_hessian (y, tx, w):
    
    s = np.zeros ((len (y), len (y)))
    
    for i in range (len (s)):
        
        v = sigmoid (tx [i].T.dot (w))
        s [i, i] = v * (1 - v)
        
    return tx.T.dot (s.dot (tx))