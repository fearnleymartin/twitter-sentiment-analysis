#Attemps to clean the tweets in order to improve the score obtained with fasttext

import numpy as np
from Files_Clement.cleaner import *
#from word2vec import generateBin, Word2Vec
from sklearn.linear_model import LogisticRegression
#from Files_Clement.helpers import trainScore, simulateTest


train_pos = '../Tweets/train_pos.txt'
train_neg = '../Tweets/train_neg.txt'
train_pos_processed = 'train_pos_processed.txt'
train_neg_processed = 'train_neg_processed.txt'
test = '../Tweets/test_data.txt'
test_processed = 'test_data_processed.txt'

pos = open (train_pos).readlines ()
neg = open (train_neg).readlines ()
test = open (test).readlines ()


pos, pos_cleaned = processTweets (pos, ['lower'], 'train_pos_clean')
pos_correct = processTweets (pos_cleaned, ['spell check'], 'train_pos_correct') [1]
neg, neg_cleaned = processTweets (neg, ['lower'], 'train_neg_clean')
neg_correct = processTweets (neg_cleaned, ['spell check'], 'train_neg_correct') [1]


#with open ('train_full_correct.txt', 'w', encoding = 'utf-8') as f:
#   f.write ('\n'.join (pos_correct) + '\n'.join (neg_correct))
with open('train_pos_correct.txt','w', encoding='utf8') as f:
    f.write('\n'.join (pos_correct))

with open('train_neg_correct.txt','w', encoding='utf8') as f:
    f.write('\n'.join(neg_correct))