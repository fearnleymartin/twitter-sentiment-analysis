"""
We train a Multi-layer Perceptron classifier using word embeddings from word2vec (gensim)
We represent each tweet as the sum of its word embeddings
"""

import numpy as np
from sklearn.cross_validation import cross_val_score
from helpers_py import export_predictions
from sklearn.neural_network import MLPClassifier
from gensim.models import word2vec

# Input file paths
train_pos = '../data/processed/train_pos.txt'
train_neg = '../data/processed/train_neg.txt'
test = '../data/processed/test_data.txt'
word2vec_model = 'models/model.bin'

# Output file paths
predictions_output = '../Results/word2vec.csv'


def load_tweets(train_pos, train_neg, test):
    """
    Loads each set of tweets from file to list
    :return: list of pos tweets, list of negative tweets, list of test tweets
    """
    with open(train_pos, encoding='utf8') as file:
        pos = file.readlines()
    with open(train_neg, encoding='utf8') as file:
        neg = file.readlines()
    with open(test) as file:
        test = file.readlines()
    return pos, neg, test


def feature_representation(model, tweets, num_features=50):
    """
    Builds feature representations of all tweets by summing feature representations of the words in the tweets
    :param model: A dictionary object {word:embedding}
    :param tweets: list of tweets (as words)
    :param num_features: number of features generated by the model
    :return: list of tweets represented a vectors
    """
    tweets_feature_repr = []
    for tweet in tweets:
        words = tweet.split(' ')  # split into words
        feature_repr = np.zeros(num_features)
        for word in words:
            if word in model:
                feature_repr += model[word]
        tweets_feature_repr.append(feature_repr.tolist ())
    return np.array(tweets_feature_repr)


def regression(positive_tweets_feature_repr, negative_tweets_feature_repr):
    """
    Build the data matrix and labels array from positive and negative tweet feature representation
    Run classifier
    :param positive_tweets_feature_repr: list of positive tweets represented as vectors
    :param negative_tweets_feature_repr: list of negative tweets represented as vectors
    :return: classifier, complet data matrix, labels
    """
    X = np.vstack((positive_tweets_feature_repr, negative_tweets_feature_repr))
    Y = np.hstack((np.ones(positive_tweets_feature_repr.shape[0]), 0*np.ones(negative_tweets_feature_repr.shape[0])))
    clf = MLPClassifier()
    clf.fit(X, Y)
    return clf, X, Y

if __name__ == '__main__':

    print("Loading Model...")
    model = word2vec.Word2Vec.load_word2vec_format(word2vec_model, binary=True)

    print("Loading Tweets...")
    positive_tweets, negative_tweets, test_tweets = load_tweets(train_pos, train_neg, test)

    print("Creating Feature Representations...")
    positive_tweets_feature_repr = feature_representation(model, positive_tweets)
    negative_tweets_feature_repr = feature_representation(model, negative_tweets)
    test_tweets_feature_repr = feature_representation(model, test_tweets)

    print("Training Multi-layer Perceptron Model...")
    clf, X, Y = regression(positive_tweets_feature_repr, negative_tweets_feature_repr)
    
    print("Computing Cross-validation Score...")
    score = cross_val_score(clf, X, Y, cv=4, scoring='accuracy')
    print("Cross-validation Score : ", score)

    print("Exporting Predictions...")
    predicted_labels = clf.predict(test_tweets_feature_repr)
    export_predictions(predicted_labels, predictions_output)
    print("Terminated")


