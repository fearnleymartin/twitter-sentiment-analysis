# -*- coding: utf-8 -*-
"""
Tokenise tweets with nltk tokeniser
Function preprocess in run on the original tweets (pos, neg and test) and gives a new file for each
"""

from nltk.tokenize.casual import TweetTokenizer
import re

full = False
tokenizer = TweetTokenizer(preserve_case=False)
dev_sample_percentage = 0.1


def preprocess(input_file, output_file, remove_duplicates=True, full=False, remove_index=False):
    """
    Tokenises tweets with nltk tweet tokeniser
    :param input_file:
    :param output_file:
    :param remove_duplicates: should be activated for train but not test data
    :param full: runs on the full data set
    :param remove_index: should be activated for test but not train data, removes the index at beginning of each tweet
    :return:
    """
    if full:
        input_file = re.sub('.txt', '_full.txt', input_file)
        output_file = re.sub('.txt', '_full.txt', output_file)
    with open(input_file, encoding='utf8') as f:
        tweets = f.readlines()
    processed_tweets = []
    for tweet in tweets:
        if remove_index:
            tweet = tweet[len(tweet.split(',')[0])+1:]   # just for test_data
        tokens = tokenizer.tokenize(tweet)
        processed_tweets.append(' '.join(tokens))
    if remove_duplicates:
        processed_tweets = set(processed_tweets)
    with open(output_file, 'w', encoding='utf8') as f:
        for tweet in processed_tweets:
            f.write(tweet + ' \n')


###################################################
# INPUT/OUTPUT FILES

# Originals files
train_pos = '../data/original/train_pos.txt'
train_neg = '../data/original/train_neg.txt'
test = '../data/original/test_data.txt'

# Tokenised / processed files
train_pos_processed = '../data/processed/train_pos_processed.txt'
train_neg_processed = '../data/processed/train_neg_processed.txt'
test_processed = '../data/processed/test_data_processed.txt'


#----------------------------------------------------------------
# PREPROCESSING : should be done only once to generate the files
# ---------------------------------------------------------------
if __name__ == "__main__":
    preprocess(train_pos, train_pos_processed, full=False)  # run on positive tweets
    preprocess(train_neg, train_neg_processed, full=False)  # run on negative tweets
    preprocess(test, test_processed, remove_duplicates=False, remove_index=True)  # run on test data





