import numpy as np
from gensim.models import word2vec
from sklearn.preprocessing import StandardScaler
from helpers import progress

def generateBin (tweetFile, outputName, nbFeatures = 50):

        sentences = word2vec.Text8Corpus (tweetFile)
        model = word2vec.Word2Vec (sentences, size = nbFeatures, min_count = 5)
        model.save_word2vec_format (outputName, binary = True)

        print ('Bin file ' + outputName + ' has been created')
        
class Word2Vec:
    
    def __tweetToVector (self, tweet):
        
        words = tweet.split(' ')
        vector = np.array ([float (0)] * self.__nbFeatures)

        for word in words:
            if word in self.__model:
                vector += self.__model [word]
                        
        return vector.tolist ()
    
    def __init__ (self, modelBin, pos, neg, nbFeatures = 50):
        
        print ('Word2Vec Instantiation')

        self.__model = word2vec.Word2Vec.load_word2vec_format (modelBin, binary = True)
        self.__nbFeatures = nbFeatures
        
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