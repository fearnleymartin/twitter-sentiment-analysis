import pandas as pd
import numpy as np

# Function displaying plotted values in order to compare features for both signals and backgrounds
# (Red is for background, Green for signals)
#

"""
# Returns
# 
#     plots (list of plot figures)
# 
"""

def compare_features (y, x):
    
    df = pd.DataFrame (np.c_ [y, x])
    df.columns = ['Prediction'] + [str (i) for i in range (x.shape [1])]
    
    ranges = []

    for i in range (df.shape [1] - 1):

        ind = df [str (i)]
        mn = np.floor (ind.min () / 10) * 10
        mx = np.ceil (ind.max () / 10) * 10
        ranges.append ([mn, mx])

    plots = []

    for i in range (df.shape [1] - 1):

        indx = str (i)
        r = ranges [i]

        plots.append (pd.DataFrame ((df [df ['Prediction'] == 1] [indx])).hist (bins = 100, range = (r [0], r [1]), layout = (1, 1),
                                                                                figsize = (20, 4), facecolor = 'green'))
        plots.append (pd.DataFrame ((df [df ['Prediction'] == 0] [indx])).hist (bins = 100, range = (r [0], r [1]), layout = (1, 1),
                                                                                figsize = (20, 4), facecolor = 'red'))
        
    return plots