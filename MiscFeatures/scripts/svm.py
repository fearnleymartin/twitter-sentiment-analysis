import numpy as np
import scripts.helpers as hlp

def hinge (x):
    
    return max (0, 1 - x)

def hinge_grad (x):
    
    if x < 0:
        return 0
    else:
        return -1
    
def svm_loss (y, tx, w, _lambda):
    
    loss = (_lambda / 2) * w.T.dot (w) [0][0]
    
    for i in range (len (y)):
        
        loss += hinge (y [i] * tx [i].T.dot (w) [0])
        
    return loss
    
def svm_grad (y, tx, w, _lambda):
                       
    grad = []
    
    for i in range (tx.shape [1]):
        
        g = _lambda * w [i]
        
        for j in range (tx.shape [0]):
            
            g += hinge_grad (y [j] * tx [j].T.dot (w)) * y [j] * tx [j, i]
            
        grad.append (g)
            
    return np.array (grad)
    
def svm_GD_step (y, tx, w, alpha, lambda_ = 0):
    
    grad = svm_grad (y, tx, w, lambda_)
    w = np.subtract (w, np.multiply (grad, alpha))
    loss = svm_loss (y, tx, w, lambda_)
    
    return loss, w
    
def svm_SGD (y, tx, max_iter = 10000, max_search_error = 200, batch_size = 1000, threshold = 1e-4, alpha = 0.01, lambda_ = 0.01, info_step = 100):
    
    w = np.zeros ((tx.shape[1], 1))
    lastLoss = float ('inf')
    
    minw = w
    minLoss = lastLoss
    lastMin = 0
    
    n_iter = 0
    while n_iter < max_iter:

        broken = False
        for minibatch_y, minibatch_tx in hlp.batch_iter (y, tx, batch_size):

            loss, w = svm_GD_step (minibatch_y, minibatch_tx, w, alpha, lambda_)
            
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
            
    print ("Final loss : ", minLoss)
    return minw