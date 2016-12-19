import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from helpers import progress

class Sentiment2Vec:

    def __init__ (self, pos, neg):
        
        print ('Sentiment2Vec Instantiation')

        self.__analyzer = SentimentIntensityAnalyzer ()

        X = []
        Y = []
        
        print ('\tConverting Train Set...')

        for tweet in progress (pos, '\t\tExtracting Features from Positive Tweets'):

            polarity = self.__analyzer.polarity_scores (tweet)

            X.append ([polarity ['pos'], polarity ['neu'], polarity ['neg'], polarity ['compound']])
            Y.append (1)

        for tweet in progress (neg, '\t\tExtracting Features from Negative Tweets'):

            polarity = self.__analyzer.polarity_scores (tweet)

            X.append ([polarity ['pos'], polarity ['neu'], polarity ['neg'], polarity ['compound']])
            Y.append (-1)

        print ('\tStandardizing...')

        X = np.array (X)

        self.__standardizer = StandardScaler ()
        self.__standardizer.fit (X)
        
        self.__train_X = self.__standardizer.transform (X)
        self.__train_Y = np.array (Y)

        print ('Terminated')
        
    def getX (self):
        return self.__train_X
        
    def getY (self):
        return self.__train_Y
    
    def convertTest (self, test):
        
        print ('Converting Test Set')
        
        test_X = []
        
        for tweet in progress (test, '\tExtracting Features from Test Tweets...'):
            polarity = self.__analyzer.polarity_scores (tweet)
            test_X.append ([polarity ['pos'], polarity ['neu'], polarity ['neg'], polarity ['compound']])

        print ('\tStandardizing...')
            
        test_X = self.__standardizer.transform (np.array (test_X))

        print ('Terminated')
        
        return test_X