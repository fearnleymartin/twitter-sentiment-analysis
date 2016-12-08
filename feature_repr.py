import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import cross_val_score
from helpers_py import export_predictions

def load_embeddings():
    return np.load('embeddings.npy')

def load_tweets(full=False):
    if full==False:
        with open('train_pos.txt','r',encoding='utf8') as file:
            pos = file.readlines()
        with open('train_neg.txt','r',encoding='utf8') as file:
            neg = file.readlines()
    else:
        with open('train_pos_full.txt','r',encoding='utf8') as file:
            pos = file.readlines()
        with open('train_neg_full.txt','r',encoding='utf8') as file:
            neg = file.readlines()
    with open('test_data.txt','r') as file:
        test=file.readlines()
    return pos, neg, test

def load_lexicon_features(full=False):
    if full==False:
        pos = np.load('pos_tweets_lexicon_features.npy')
        neg = np.load('neg_tweets_lexicon_features.npy')
    else:
        pos = np.load('pos_tweets_full_lexicon_features.npy')
        neg = np.load('neg_tweets_full_lexicon_features.npy')

    test = np.load('test_tweets_lexicon_features.npy')

    return pos, neg, test

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

def feature_representation_v2(embeddings, tweets, vocab_dict):
    """
    build feature representations of tweets by summing feature representations of the words in the tweets
    :param embeddings:
    :param positive_tweets:
    :param negative_tweets:
    :param vocab_dict:
    :return:
    """
    tweets_feature_repr = np.zeros((len(tweets), 20))
    for index, tweet in enumerate(tweets):
        words = tweet.split(' ')
        feature_repr = np.zeros(20)
        for word in words:
            if word in vocab_dict.keys():
                feature_repr += embeddings[vocab_dict[word]]
        tweets_feature_repr[index] = feature_repr

    return tweets_feature_repr

def concatenate_features(feature_1,feature_2):
    return np.concatenate((feature_1,feature_2),axis=1)

def regression(positive_tweets_feature_repr, negative_tweets_feature_repr):
    """
    Build the data matrix and labels array from positive and negative tweet feature representation
    Run logistic regression classifier
    :param positive_tweets_feature_repr:
    :param negative_tweets_feature_repr:
    :return:
    """
    X = np.vstack((positive_tweets_feature_repr, negative_tweets_feature_repr))
    Y = np.hstack((np.ones(positive_tweets_feature_repr.shape[0]), -1*np.ones(negative_tweets_feature_repr.shape[0])))
    clf = LogisticRegression()
    clf.fit(X, Y)
    return clf, X, Y

def cross_validation(clf, X, Y):
    return cross_val_score(clf, X, Y, cv=5, scoring='accuracy')


if __name__ == '__main__':
    '''
    embeddings = load_embeddings()
    print("Embeddings loaded")
    positive_tweets, negative_tweets,test_tweets = load_tweets(full=True)
    vocab_dict = load_vocab()
    positive_tweets_feature_repr = feature_representation_v2(embeddings, positive_tweets, vocab_dict)
    negative_tweets_feature_repr = feature_representation_v2(embeddings,negative_tweets,vocab_dict)
    test_tweets_feature_repr = feature_representation_v2(embeddings,test_tweets,vocab_dict)
    print("First feature representation achieved")
    '''
    pos_tweets_lexicon_features,neg_tweets_lexicon_features,test_tweets_lexicon_features=load_lexicon_features(full=True)
    print("Lexicon features loaded")
    '''
    pos_tweets_features = concatenate_features(positive_tweets_feature_repr,pos_tweets_lexicon_features)
    neg_tweets_features = concatenate_features(negative_tweets_feature_repr,neg_tweets_lexicon_features)
    test_tweets_features = concatenate_features(test_tweets_feature_repr,test_tweets_lexicon_features)
    print("Features concatenated")
    '''
    clf, X, Y = regression(pos_tweets_lexicon_features, neg_tweets_lexicon_features)
    score = cross_validation(clf, X, Y)
    print(score)
    predicted_labels=clf.predict(test_tweets_lexicon_features)
    print(predicted_labels)
    export_predictions(predicted_labels,'submission_lexicon_only_features')



