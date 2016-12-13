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


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x