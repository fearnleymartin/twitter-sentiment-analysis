import csv
import numpy as np
import time
import datetime

def exportPredictions (predictions, name = 'kaggle_submission'):
    
    if len (predictions) != 10000:
        raise ValueError ('There must be exactly 10,000 predictions to submit to kaggle')
        
    ids = range (1, 10001)
    
    with open (name + '.csv', 'w', newline='') as csvfile:
        
        fieldnames = ['Id', 'Prediction']
        
        writer = csv.DictWriter (csvfile, delimiter = ",", fieldnames = fieldnames)
        writer.writeheader()
        
        for r1, r2 in zip (ids, predictions):
            writer.writerow ({'Id' : int (r1), 'Prediction' : int (r2)})
            
            
            
            
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

def trainScore (X, Y, classifier):
    
    r = 0
    n = X.shape [0]
    
    for i in progress (range (n), 'Computing Train Score'):
        
        if classifier.predict (X [i, :].reshape (1, -1)) [0] == Y [i]:
            r += 1
            
    return 1.0 * r / X.shape [0]

def progress (iterable, process_name):
    
    i = 0
    n = len (iterable)
    l = 0
    print (process_name + ' [' + '-' * 20 + ']    (0%)    ETA : ', end = '', flush = True)
    
    remaining = -1
    then = time.time ()
    
    for ite in iterable:
        
        yield ite
        
        i += 1
        
        if (i * 100) // n > l:
            
            l += 1
            l5 = l // 5
            
            now = time.time ()
            remaining = (now - then) * (100 - l)
            then = now
            eta = (datetime.datetime.now () + datetime.timedelta (seconds=int (remaining))).strftime ('%H:%M:%S')
            sep = '    '
            
            if l < 100:
                if l < 10:
                    sep += ' '
                sep += ' '
            
            print ('\r', end = '')
            print (process_name + ' [' + '#' * l5 + '-' * (20 - l5) + ']    (' + str (l) + '%)' + sep + 'ETA : ' + eta, end = '', flush = True)
            
    print ('\n', end = '', flush = True)

def predict (test_X, classifier):
    
    prediction = []
    n = test_X.shape [0]
    
    for i in progress (range (n), 'Computing Predictions'):
        
        prediction.append (classifier.predict (test_X [i, :].reshape (1, -1)) [0])
            
    return prediction

def testScore (test_X, classifier):
    
    Y = [1] * 5000 + [-1] * 5000
    
    r = 0
    n = test_X.shape [0]
    
    for i in progress (range (n), 'Computing Test Score'):
        
        if classifier.predict (test_X [i, :].reshape (1, -1)) [0] == Y [i]:
            r += 1
            
    return 1.0 * r / test_X.shape [0]