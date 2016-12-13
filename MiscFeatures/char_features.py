import numpy as np
from sklearn.preprocessing import StandardScaler
import operator

def generateCharList (pos, neg):
    
    chars = {}
    
    fp = open (pos)
    line = fp.readline ()
    
    while line:
        
        for char in line:
            if char not in chars:
                chars [char] = 1
            else:
                chars [char] += 1
                
        line = fp.readline ()
        
    fp.close ()
    
    fp = open (neg)
    line = fp.readline ()
    
    while line:
        
        for char in line:
            if char not in chars:
                chars [char] = 1
            else:
                chars [char] += 1
                
        line = fp.readline ()
        
    fp.close ()
    
    return chars

def tweetToVector (tweet, chars):
    
    vector = [0] * len (chars)
    
    for char in tweet:
        if char in chars:
            vector [chars [char]] += 1
        
    return vector

def generateCharFeatures (pos, neg, keep = 1.0):
    
    print ('Generating char list...')
    
    characters = generateCharList (pos, neg)
    characters = dict((k, v) for k, v in characters.items() if v != 200000)
    characters = sorted (characters.items(), key = operator.itemgetter (1))
    characters.reverse ()
    
    print ('Filtering char list...')
    
    i = 0
    filter_characters = {}
    
    for k, v in characters:
        
        i += 1
        
        if i > keep * len (characters):
            break
            
        filter_characters [k] = v
    
    chars = {}
    i = 0
    to_print = '['

    for k in filter_characters.keys():
        
        chars [k] = i
        
        to_print += '\'' + k + '\''
        
        i += 1
        
        if i == len (filter_characters):
            to_print += ']'
        else:
            to_print += ', '
    
    X = []
    Y = []
    
    print ('Extracting features from positive tweets...')
    
    fp = open (pos, encoding='utf8')
    lines = fp.readlines ()
    done = 0
    i = 0
    n = len (lines)
    
    for line in lines:
        
        X.append (tweetToVector (line, chars))
        Y.append (1)
        
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
        
        X.append (tweetToVector (line, chars))
        Y.append (-1)
        
        i += 1
        
        if (i * 10) // n > done:
            done += 1
            print (str (done) + '0% Done')
        
    fp.close ()
    
    print ('Standardizing...')
    
    X = np.array (X)
    
    scaler = StandardScaler ()
    scaler.fit (X)
    tX = scaler.transform (X)
    
    print ('Done')
    
    print ('Featured chars : ', to_print)
    
    return tX, np.array (Y)