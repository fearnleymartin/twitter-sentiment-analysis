# -*- coding: utf-8 -*-
"""  Some helper functions for project 1  """

import csv
import numpy as np


# Functions useful to load csv data direcly into manupilable data.
# 
#     data_path is the path of the file to read
#     sub_sample should be set to True if we want only 50 entries, to reduce time execution and buffer
# 
    
"""
# Returns
# 
#     y (class labels)
#     tx (features)
#     IDs (event IDs)
#
"""

def load_csv_data (data_path, sub_sample = False):
    
    # Read the csv file into input data
    y = np.genfromtxt (data_path, delimiter = ",", skip_header = 1, dtype = str, usecols = 1)
    x = np.genfromtxt (data_path, delimiter = ",", skip_header = 1)
    
    # Extract IDs of events
    ids = x [:, 0].astype (np.int)
    
    # Extract other data
    input_data = x [:, 2:]

    # Convert class labels from strings to binary (-1,1)
    yb = np.ones (len (y))
    yb [np.where(y=='b')] = -1
    
    # Sub-sample if needed
    if sub_sample:
        yb = yb [::50]
        input_data = input_data [::50]
        ids = ids [::50]

    return yb, input_data, ids


# Function calcilating the predictions given an input x and a prediction function w
# 
#     x is the input
#     w is the prediction function (or weight)
# 
    
"""
# Returns
# 
#     y_pred (prediction output)
#
"""

def predict_labels (x, w):
    
    # Apply the prediction function
    y_pred = np.dot(x, w)
    
    # Normalize the output
    y_pred [np.where(y_pred <= 0)] = -1
    y_pred [np.where(y_pred > 0)] = 1
    
    return y_pred


# Function useful to create a correct output for Kaggle (Competition purposes)
# 
#     name is the name of the output csv file
#     
#     IDs is the event IDs from the test data
#     y_pred is the predicted output, obtained using ML
#     

def create_csv_submission (name, ids, y_pred):
    
    with open (name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter (csvfile, delimiter = ",", fieldnames = fieldnames)
        writer.writeheader ()
        for r1, r2 in zip (ids, y_pred):
            writer.writerow ({'Id': int (r1),'Prediction': int (r2)})
