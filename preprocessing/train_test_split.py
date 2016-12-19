import re
import numpy as np


def separate_crossval(input_file_pos,output_file_pos,input_file_neg,output_file_neg,output_file_crossval,percentage,full=False):
    if full:
        input_file_pos = re.sub('.txt', '_full.txt', input_file_pos)
        output_file_pos = re.sub('.txt', '_full.txt', output_file_pos)
        input_file_neg = re.sub('.txt', '_full.txt', input_file_neg)
        output_file_neg = re.sub('.txt', '_full.txt', output_file_neg)
        output_file_crossval = re.sub('.txt', '_full.txt', output_file_crossval)

    with open(input_file_pos, encoding='utf8') as file_pos:
        tweets_pos = file_pos.readlines()
    with open(input_file_neg, encoding='utf8') as file_neg:
        tweets_neg=file_neg.readlines()
    dev_sample_index_pos = -1*int(percentage*float(len(tweets_pos)))
    dev_sample_index_neg = -1 * int(percentage * float(len(tweets_neg)))
    pos_train, pos_dev = tweets_pos[:dev_sample_index_pos], tweets_pos[dev_sample_index_pos:]
    neg_train, neg_dev = tweets_neg[:dev_sample_index_neg], tweets_neg[dev_sample_index_neg:]

    with open(output_file_pos, 'w', encoding='utf8') as f:
        for tweet in pos_train:
            f.write(tweet)

    with open(output_file_neg, 'w', encoding='utf8') as f:
        for tweet in neg_train:
            f.write(tweet)


    np.random.seed(10)
    #tweets_crossval = np.vstack((pos_dev, neg_dev))
    y = np.hstack((np.ones(len(pos_dev)), -1 * np.ones(len(neg_dev))))
    #shuffle_indices = np.random.permutation(np.arange(len(tweets_crossval)))
    #tweets_crossval=tweets_crossval[shuffle_indices]
    #y=y[shuffle_indices]

    with open(output_file_crossval, 'w', encoding='utf8') as f:
        for tweet in pos_dev:
            f.write(tweet)
        for tweet in neg_dev:
            f.write(tweet)

    np.save('y_CV', y)
    print('Number of items in the Test_CV set : ', len(y))


#-------------------------------------------------------------------------------
# CROSS VALIDATION : choose the percentage of the CV
#---------------------------------------------------------------------------------

train_pos_processed = 'train_pos_processed.txt'
train_neg_processed = 'train_neg_processed.txt'

separate_crossval(train_pos_processed,'train_pos_processed_CV.txt', train_neg_processed,'train_neg_processed_CV.txt','test_CV.txt', 0.05, full=False)

