"""
Trains fasttext model and evaluates on test_data
Specify input tweets file (already formatted for running with fasttext
"""

import os

# input paths
dir_path = '../../../vagrant/fasttext/'
input_tweets_file = dir_path + '../data/fasttext/train_fasttext_full.txt'  # Should be preprocessed and correctly formatted (see readme)

# intermediary paths (probably don't need to modify)
model_path = dir_path + 'models/model'
test_path = dir_path + '../data/processed/test_data_processed.txt'

# output path
test_predictions_path = dir_path + 'results/results.txt'

# params
wordNgrams = 2
num_epochs = 5
dim = 100
ws = 5
learning_rate = 0.1


def train(input_file, model_path, wordNgrams, num_epochs, dim):
    """
    Trains a classification model using fasttext, outputs a model file
    :param input_file: input tweet file to use, should be proprocessed and correctly formatted (see readme)
    :param model_path: the output path for the binary model file
    :param wordNgrams: max length of word ngram
    :param num_epochs: number of epochs
    :param dim: size of word vectors
    :return: None
    """
    train_command = './fasttext supervised -input {} -output {} -wordNgrams {} -epoch {} -dim {} -ws {} -lr {}'.format(input_file,
                                                                                                     model_path,
                                                                                                     wordNgrams,
                                                                                                     num_epochs, dim, ws, learning_rate)
    os.system(train_command)

def eval(model_path, test_path, test_predictions_path):
    """
    Predicts labels for test data
    :param model_path: binary model file to use
    :param test_path: test tweets to use
    :param test_predictions_path: output path for predictions
    :return: None
    """
    eval_command = './fasttext predict {}.bin {} > {}'.format(model_path, test_path, test_predictions_path)
    os.system(eval_command)


def accuracy(test_predictions_path, test_labels_path):
    """
    Calculates accuracy of model predictions compared with correct labels
    :param test_predictions_path: path for predictions
    :param test_labels_path: path for correct predictions
    :return:
    """
    with open(test_predictions_path) as f:
        predictions = f.readlines()
    with open(test_labels_path) as f:
        correct_labels = f.readlines()

    total = 0
    for label1, label2 in zip(predictions, correct_labels):
        if label1 == label2:
            total += 1
    acc = float(total) / float(len(predictions))
    print('accuracy is {}'.format(acc))
    return acc

if __name__ == "__main__":
    train(input_tweets_file, model_path, wordNgrams, num_epochs, dim)
    eval(model_path, test_path, test_predictions_path)