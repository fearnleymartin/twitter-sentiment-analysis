import numpy as np
from sklearn.cross_validation import cross_val_score
from helpers_py import export_predictions
from sklearn.ensemble import RandomForestClassifier


# Input file paths
train_pos = '../data/processed/train_pos_processed_full.txt'
train_neg = '../data/processed/train_neg_processed_full.txt'
test = '../data/processed/test_data_processed.txt'
embeddings = 'embeddings.npy'
vocabulary = 'vocab_cut.txt'

# Output file paths
predictions_output = 'results/predictions.txt'


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


def load_vocab():
    """
    Build dictionary of vocabulary to recover index of word in embeddings
    ex: vocab_dict[the] = 3
    :return: {word:index,...}
    """
    vocab_dict = {}
    with open(vocabulary) as file:
        vocab = file.readlines()
    for index, word in enumerate(vocab):
        vocab_dict[word] = index
    return vocab_dict


def feature_representation(embeddings, tweets, vocab_dict):
    """
    build feature representations of tweets by summing feature representations of the words in the tweets
    :param embeddings: dict {word_index, vector_representation_of_word}
    :param tweets: list of tweets
    :param vocab_dict: lookup dictionary for getting index from word
    :return:
    """
    tweets_feature_repr = np.zeros((len(tweets), 20))
    for index, tweet in enumerate(tweets):
        words = tweet.split(' ')  # split into words
        feature_repr = np.zeros(20)
        for word in words:
            if word in vocab_dict.keys():
                feature_repr += embeddings[vocab_dict[word]]
        tweets_feature_repr[index] = feature_repr
    return tweets_feature_repr


def regression(positive_tweets_feature_repr, negative_tweets_feature_repr):
    """
    Build the data matrix and labels array from positive and negative tweet feature representation
    Run logistic regression classifier
    :param positive_tweets_feature_repr: list of positive tweets represented as vectors
    :param negative_tweets_feature_repr: list of negative tweets represented as vectors
    :return: classifier, data matrix, labels
    """
    X = np.vstack((positive_tweets_feature_repr, negative_tweets_feature_repr))
    Y = np.hstack((np.ones(positive_tweets_feature_repr.shape[0]), 0*np.ones(negative_tweets_feature_repr.shape[0])))
    clf = RandomForestClassifier()
    clf.fit(X, Y)
    return clf, X, Y


if __name__ == '__main__':
    embeddings = np.load(embeddings)
    positive_tweets, negative_tweets, test_tweets = load_tweets(train_neg, train_neg, test)
    vocab_dict = load_vocab()
    positive_tweets_feature_repr = feature_representation(embeddings, positive_tweets, vocab_dict)
    negative_tweets_feature_repr = feature_representation(embeddings, negative_tweets, vocab_dict)
    test_tweets_feature_repr = feature_representation(embeddings, test_tweets, vocab_dict)
    print("feature representation created")

    clf, X, Y = regression(positive_tweets_feature_repr, negative_tweets_feature_repr)
    score = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print(score)

    predicted_labels = clf.predict(test_tweets_feature_repr)
    print(predicted_labels)
    export_predictions(predicted_labels, predictions_output)



