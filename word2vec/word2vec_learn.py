import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import cross_val_score
from helpers_py import export_predictions
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from gensim.models import word2vec

def load_tweets(full=False, cleaned=True):
    if not full:
        if cleaned:
            with open('train_pos_cleaned.txt', encoding='utf8') as file:
                pos = file.readlines()
            with open('train_neg_cleaned.txt', encoding='utf8') as file:
                neg = file.readlines()
        else:
            with open('train_pos.txt', encoding='utf8') as file:
                pos = file.readlines()
            with open('train_neg.txt', encoding='utf8') as file:
                neg = file.readlines()
    else:
        if cleaned:
            with open('train_pos_full_cleaned.txt', encoding='utf8') as file:
                pos = file.readlines()
            with open('train_neg_full_cleaned.txt', encoding='utf8') as file:
                neg = file.readlines()
        else:
            with open('train_pos_full.txt', encoding='utf8') as file:
                pos = file.readlines()
            with open('train_neg_full.txt', encoding='utf8') as file:
                neg = file.readlines()
    with open('test_data.txt') as file:
        test = file.readlines()
    return pos, neg, test


def load_lexicon_features(full=False):
    if not full:
        pos = np.load('pos_tweets_lexicon_features.npy')
        neg = np.load('neg_tweets_lexicon_features.npy')
    else:
        pos = np.load('pos_tweets_full_lexicon_features.npy')
        neg = np.load('neg_tweets_full_lexicon_features.npy')

    test = np.load('test_tweets_lexicon_features.npy')

    return pos, neg, test


def feature_representation_v2(model, tweets, num_features=50):
    """
    build feature representations of tweets by summing feature representations of the words in the tweets
    :param embeddings:
    :param tweets:
    :param vocab_dict:
    :return:
    """

    tweets_feature_repr = np.zeros((len(tweets), num_features))
    for index, tweet in enumerate(tweets):
        words = tweet.split(' ')  # split into words
        feature_repr = np.zeros(num_features)
        for word in words:
            if word in model:
                feature_repr += model[word]
        tweets_feature_repr[index] = feature_repr

    return tweets_feature_repr


def concatenate_features(feature_1, feature_2):
    return np.concatenate((feature_1, feature_2), axis=1)


def regression(positive_tweets_feature_repr, negative_tweets_feature_repr):
    """
    Build the data matrix and labels array from positive and negative tweet feature representation
    Run logistic regression classifier
    :param positive_tweets_feature_repr:
    :param negative_tweets_feature_repr:
    :return:
    """
    X = np.vstack((positive_tweets_feature_repr, negative_tweets_feature_repr))
    Y = np.hstack((np.ones(positive_tweets_feature_repr.shape[0]), 0*np.ones(negative_tweets_feature_repr.shape[0])))
    # clf = LogisticRegression()
    # clf = RandomForestClassifier(n_estimators=500, max_features='log2', max_depth=8, min_samples_leaf=2)
    clf = RandomForestClassifier()

    # clf.fit(X, Y)
    return clf, X, Y


def cross_validation(clf, X, Y):
    return cross_val_score(clf, X, Y, cv=4, scoring='accuracy')


if __name__ == '__main__':
    model = word2vec.Word2Vec.load_word2vec_format('word2vec.model.bin', binary=True)
    print("model loaded")
    positive_tweets, negative_tweets, test_tweets = load_tweets(full=False, cleaned=True)
    print("tweets loaded")
    positive_tweets_feature_repr = feature_representation_v2(model, positive_tweets)
    negative_tweets_feature_repr = feature_representation_v2(model, negative_tweets)
    test_tweets_feature_repr = feature_representation_v2(model, test_tweets)
    print("First feature representation achieved")

    # pos_tweets_lexicon_features, neg_tweets_lexicon_features, test_tweets_lexicon_features = load_lexicon_features(full=True)
    # print("Lexicon features loaded")
    # pos_tweets_features = concatenate_features(positive_tweets_feature_repr, pos_tweets_lexicon_features)
    # neg_tweets_features = concatenate_features(negative_tweets_feature_repr, neg_tweets_lexicon_features)
    # test_tweets_features = concatenate_features(test_tweets_feature_repr, test_tweets_lexicon_features)
    # print("Features concatenated")

    clf, X, Y = regression(positive_tweets_feature_repr, negative_tweets_feature_repr)
    score = cross_validation(clf, X, Y)
    print(score)

    # predicted_labels = clf.predict(test_tweets_feature_repr)
    # print(predicted_labels)
    # export_predictions(predicted_labels, 'submission_gensim_full')



