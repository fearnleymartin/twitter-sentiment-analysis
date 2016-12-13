import csv
import numpy as np

def export_predictions (predictions, name = 'kaggle_submission'):
    
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

def trainScore (Y, X, classifier):
    
    r = 0
    done = 0
    k = 0
    n = X.shape [0]
    
    for i in range (n):
        
        if classifier.predict (X [i, :].reshape (1, -1)) [0] == Y [i]:
            r += 1
        
        k += 1
        
        if (k * 10) // n > done:
            done += 1
            print (str (done) + '0% Done')
            
    return 1.0 * r / X.shape [0]