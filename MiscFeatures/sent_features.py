import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler

def generateSentFeatures (pos, neg):
    
    analyzer = SentimentIntensityAnalyzer ()
    
    X = []
    
    print ('Extracting features from positive tweets...')
    
    fp = open (pos, encoding='utf8')
    lines = fp.readlines ()
    done = 0
    i = 0
    n = len (lines)
    
    for line in lines:
        
        polarity = analyzer.polarity_scores (line)
        
        X.append ([polarity ['neg'], polarity ['neu'], polarity ['pos'], polarity ['compound']])
        
        i += 1
        
        if (i * 10) // n > done:
            done += 1
            print (str (done) + '0% Done')
        
    fp.close ()
    
    print ('Extracting features from negative tweets...')
    
    fp = open (pos, encoding='utf8')
    lines = fp.readlines ()
    done = 0
    i = 0
    n = len (lines)
    
    for line in lines:
        
        polarity = analyzer.polarity_scores (line)
        
        X.append ([polarity ['neg'], polarity ['neu'], polarity ['pos'], polarity ['compound']])
        
        i += 1
        
        if (i * 10) // n > done:
            done += 1
            print (str (done) + '0% Done')
        
    fp.close ()
    
    print ('Standardizing...')
    
    X = np.array (X)
    
    scaler = StandardScaler ()
    scaler.fit (X)
    X = scaler.transform (X)
    
    print ('Done')
    
    return X