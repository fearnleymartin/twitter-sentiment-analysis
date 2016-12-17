import re

full = False

train_pos_processed = 'train_pos_processed.txt'
train_neg_processed = 'train_neg_processed.txt'
train_fasttext = 'train_fasttext.txt'
train_fasttext_no_label = 'train_fasttext_no_label.txt'
# train_pos_fasttext = 'train_pos_processed_fasttext.txt'
# train_neg_fasttext = 'train_neg_processed_fasttext.txt'

def preprocess(input_file_pos, input_file_neg, output_file, full=False, label=True):
    """
    Formats tweets to use with fasttext, add label at beginning of each tweet
    :param input_file_pos:
    :param input_file_neg:
    :param output_file: combines pos and neg to a single file
    :param full: choose full data set or not
    :param label: for test data set, we don't want to add a label
    :return:
    """
    if full:
        input_file_pos = re.sub('.txt', '_full.txt', input_file_pos)
        input_file_neg = re.sub('.txt', '_full.txt', input_file_neg)
        output_file = re.sub('.txt', '_full.txt', output_file)
    with open(output_file, 'w', encoding='utf8') as of:
        with open(input_file_pos, encoding='utf8') as f:
            for pos_tweet in f:
                if label:
                    of.write('__label__POS, ' + pos_tweet)
                else:
                    of.write(pos_tweet)
        with open(input_file_neg, encoding='utf8') as f:
            for neg_tweet in f:
                if label:
                    of.write('__label__NEG, ' + neg_tweet)
                else:
                    of.write(neg_tweet)

def postprocess(input_file, output_file, full=False):
    """
    Converts from fasttext output format to kaggle submission format
    :param input_file:
    :param output_file:
    :param full:
    :return:
    """
    if full:
        input_file = re.sub('.txt', '_full.txt', input_file)
        output_file = re.sub('.txt', '_full.txt', output_file)
    with open(output_file, 'w', encoding='utf8') as of:
        of.write('Id,Prediction\n')
        with open(input_file, encoding='utf8') as f:
            for index, line in enumerate(f):
                if line == '__label__NEG,\n':
                    of.write(str(index+1)+','+'-1\n')
                elif line == '__label__POS,\n':
                    of.write(str(index+1)+','+'1\n')
                else:
                    raise Exception('Unknown label', line)


# Make sure to select correct lines to run

preprocess(train_pos_processed, train_neg_processed, train_fasttext_no_label, label=False, full=True)
# preprocess(train_pos_processed, train_neg_processed, train_fasttext, full=True)
# postprocess('results_fasttext.txt', 'results_fasttext_processed.txt', full=True)
