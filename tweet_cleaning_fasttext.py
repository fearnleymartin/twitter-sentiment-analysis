#Attemps to clean the tweets in order to improve the score obtained with fasttext

import numpy as np
from Files_Clement.cleaner import *
#from word2vec import generateBin, Word2Vec
from sklearn.linear_model import LogisticRegression
#from Files_Clement.helpers import trainScore, simulateTest


full = True;
#Blank data
train_pos = '../Tweets/train_pos.txt'
train_neg = '../Tweets/train_neg.txt'
test = '../Tweets/test_data.txt'

if full:
    train_pos = re.sub('.txt', '_full.txt', train_pos)
    train_neg = re.sub('.txt', '_full.txt', train_neg)


with open(train_pos, encoding='utf8') as f:
    pos = f.readlines()
with open(train_neg, encoding='utf8') as f:
    neg = f.readlines()
with open(test, encoding='utf8') as f:
    test = f.readlines()



pos, pos_cleaned = processTweets (pos, ['lower'], 'train_pos_clean')
pos_correct = processTweets (pos_cleaned, ['spell check'], 'train_pos_correct') [1]
neg, neg_cleaned = processTweets (neg, ['lower'], 'train_neg_clean')
neg_correct = processTweets (neg_cleaned, ['spell check'], 'train_neg_correct') [1]
test, test_cleaned = processTweets (test, ['lower'], 'test_clean')
test_correct = processTweets (test_cleaned, ['spell check'], 'test_correct') [1]


with open('train_pos_correct.txt','w', encoding='utf8') as f:
    for tweet in pos_correct:
        tweet = re.sub("\n", '', tweet)
        f.write(tweet + '\n')

with open('train_neg_correct.txt','w', encoding='utf8') as f:
    for tweet in neg_correct:
        tweet = re.sub("\n", '', tweet)
        f.write(tweet + '\n')

with open('test_correct.txt','w', encoding='utf8') as f:
    for tweet in test_correct:
        tweet = re.sub("\n", '', tweet)
        f.write(tweet + '\n')