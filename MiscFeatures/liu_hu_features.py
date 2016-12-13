import numpy as np
from nltk.corpus import opinion_lexicon
from nltk.tokenize import treebank
from sklearn.preprocessing import StandardScaler

def generateLiuHuFeatures (pos, neg):
    
    tokenizer = treebank.TreebankWordTokenizer()
    
    X = []
    
    print ('Extracting features from positive tweets...')
    
    fp = open (pos, encoding='utf8')
    lines = fp.readlines ()
    done = 0
    i = 0
    n = len (lines)
    
    for line in lines:
        
        tokenized_sent = [word.lower() for word in tokenizer.tokenize (line)]
        
        pos = 0
        neu = 0
        neg = 0
        
        for word in tokenized_sent:
            if word in opinion_lexicon.positive ():
                pos += 1
            elif word in opinion_lexicon.negative ():
                neg += 1
            else:
                neu += 1
                
        X.append ([pos, neu, neg])
        
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
        
        tokenized_sent = [word.lower() for word in tokenizer.tokenize (line)]
        
        pos = 0
        neu = 0
        neg = 0
        
        for word in tokenized_sent:
            if word in opinion_lexicon.positive ():
                pos += 1
            elif word in opinion_lexicon.negative ():
                neg += 1
            else:
                neu += 1
                
        X.append ([pos, neu, neg])
        
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