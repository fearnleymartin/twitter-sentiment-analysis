import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import cross_val_score


def load_embeddings():
    return np.load('embeddings.npy')

def load_tweets():
    with open('train_pos.txt','r') as file:
        pos = file.readlines()
    with open('train_neg.txt','r') as file:
        neg = file.readlines()
    return pos, neg

def load_vocab():
    """
    build dictionnary of vocabulary to recover index of word in embeddings
    ex: vocab_dict[the] = 3
    """
    vocab_dict = {}
    with open('vocab_cut.txt') as file:
        vocab = file.readlines()
    for index, word in enumerate(vocab):
        vocab_dict[word] = index
    return vocab_dict


def feature_representation(embeddings, positive_tweets, negative_tweets, vocab_dict):
    """
    build feature representations of tweets by summing feature representations of the words in the tweets
    :param embeddings:
    :param positive_tweets:
    :param negative_tweets:
    :param vocab_dict:
    :return:
    """
    positive_tweets_feature_repr = np.zeros((len(positive_tweets), 20))
    for index, tweet in enumerate(positive_tweets):
        words = tweet.split(' ')
        feature_repr = np.zeros(20)
        for word in words:
            if word in vocab_dict.keys():
                feature_repr += embeddings[vocab_dict[word]]
        positive_tweets_feature_repr[index] = feature_repr

    negative_tweets_feature_repr = np.zeros((len(negative_tweets), 20))
    for index, tweet in enumerate(negative_tweets):
        words = tweet.split(' ')
        feature_repr = np.zeros(20)
        for word in words:
            if word in vocab_dict.keys():
                feature_repr += embeddings[vocab_dict[word]]
        negative_tweets_feature_repr[index] = feature_repr
    return positive_tweets_feature_repr, negative_tweets_feature_repr

def regression(positive_tweets_feature_repr, negative_tweets_feature_repr):
    """
    Build the data matrix and labels array from positive and negative tweet feature representation
    Run logistic regression classifier
    :param positive_tweets_feature_repr:
    :param negative_tweets_feature_repr:
    :return:
    """
    X = np.vstack((positive_tweets_feature_repr, negative_tweets_feature_repr))
    Y = np.hstack((np.ones(positive_tweets_feature_repr.shape[0]), np.zeros(negative_tweets_feature_repr.shape[0])))
    clf = LogisticRegression()
    clf.fit(X, Y)
    return clf, X, Y

def cross_validation(clf, X, Y):
    return cross_val_score(clf, X, Y, cv=5, scoring='accuracy')


if __name__ == '__main__':
    embeddings = load_embeddings()
    positive_tweets, negative_tweets = load_tweets()
    vocab_dict = load_vocab()
    positive_tweets_feature_repr, negative_tweets_feature_repr = feature_representation(embeddings, positive_tweets, negative_tweets, vocab_dict)
    clf, X, Y = regression(positive_tweets_feature_repr, negative_tweets_feature_repr)
    score = cross_validation(clf, X, Y)
    print(score)

