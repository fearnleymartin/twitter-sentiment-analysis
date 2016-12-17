import numpy as np
from sklearn.preprocessing import StandardScaler
from helpers import progress

class Char2Vec:
    
    def __expandCharDictionary (self, tweets):
        
        for tweet in tweets:

            for char in tweet:
                if char not in self.__chars:
                    self.__chars [char] = len (self.__chars)

    def __createCharDictionary (self, pos, neg):
        
        self.__chars = {}
        self.__expandCharDictionary (pos)
        self.__expandCharDictionary (neg)

    def __tweetToVector (self, tweet):

        vector = [0] * len (self.__chars)

        for char in tweet:
            if char in self.__chars:
                vector [self.__chars [char]] += float (1)

        return vector

    def __init__ (self, pos, neg):
        
        print ('Char2Vec Instantiation')

        print ('\tCreating Char Dictionary...')

        self.__createCharDictionary (pos, neg)
        
        print ('\tConverting Train Set...')

        X = []
        Y = []

        for tweet in progress (pos, '\t\tExtracting Features from Positive Tweets...'):

            X.append (self.__tweetToVector (tweet))
            Y.append (1)

        for tweet in progress (neg, '\t\tExtracting Features from Negative Tweets...'):

            X.append (self.__tweetToVector (tweet))
            Y.append (-1)

        print ('\tStandardizing...')

        X = np.array (X)

        self.__standardizer = StandardScaler ()
        self.__standardizer.fit (X)
        
        self.__train_X = self.__standardizer.transform (X)
        self.__train_Y = np.array (Y)

        print ('Terminated')
        
    def viewCharFeatures (self):
        
        view = [''] * len (self.__chars)
        
        for k, v in self.__chars.items ():
            view [v] = k
            
        print ('In order : ', view)
        
    def getX (self):
        return self.__train_X
        
    def getY (self):
        return self.__train_Y
    
    def convertTest (self, test):
        
        print ('Converting Test Set')
        
        test_X = []
        
        for tweet in progress (test, '\tExtracting Features from Test Tweets...'):
            test_X.append (self.__tweetToVector (tweet))

        print ('\tStandardizing...')
            
        test_X = self.__standardizer.transform (np.array (test_X))

        print ('Terminated')
        
        return test_X