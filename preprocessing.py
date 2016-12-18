from nltk.tokenize.casual import TweetTokenizer
import re
import numpy as np

full = False
tokenizer = TweetTokenizer(preserve_case=False)
dev_sample_percentage=0.1


def preprocess(input_file, output_file, remove_duplicates=True, full=False, remove_index=False):
    """
    Tokenises tweets with nltk tweet tokensier
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

def separate_crossval(input_file_pos,output_file_pos,input_file_neg,output_file_neg,output_file_crossval,percentage,full=False):
    if full:
        input_file_pos = re.sub('.txt', '_full.txt', input_file_pos)
        output_file_pos = re.sub('.txt', '_full.txt', output_file_pos)
        input_file_neg = re.sub('.txt', '_full.txt', input_file_neg)
        output_file_neg = re.sub('.txt', '_full.txt', output_file_neg)
        output_file_crossval = re.sub('.txt', '_full.txt', output_file_crossval)


    with open(input_file_pos, encoding='utf8') as file_pos:
        tweets_pos=file_pos.readlines()
    with open(input_file_neg,encoding='utf8') as file_neg:
        tweets_neg=file_neg.readlines()
    dev_sample_index_pos = -1*int(percentage*float(len(tweets_pos)))
    dev_sample_index_neg = -1 * int(percentage * float(len(tweets_neg)))
    pos_train,pos_dev = tweets_pos[:dev_sample_index_pos], tweets_pos[dev_sample_index_pos:]
    neg_train,neg_dev = tweets_neg[:dev_sample_index_neg], tweets_neg[dev_sample_index_neg:]

    with open(output_file_pos,'w',encoding='utf8') as f:
        for tweet in pos_train:
            f.write(tweet)

    with open(output_file_neg,'w',encoding='utf8') as f:
        for tweet in neg_train:
            f.write(tweet)


    np.random.seed(10)
    #tweets_crossval = np.vstack((pos_dev, neg_dev))
    y = np.hstack((np.ones(len(pos_dev)), -1 * np.ones(len(neg_dev))))
    #shuffle_indices = np.random.permutation(np.arange(len(tweets_crossval)))
    #tweets_crossval=tweets_crossval[shuffle_indices]
    #y=y[shuffle_indices]

    with open(output_file_crossval,'w',encoding='utf8') as f:
        for tweet in pos_dev:
            f.write(tweet)
        for tweet in neg_dev:
            f.write(tweet)

    np.save('y_CV',y)
    print('Number of items in the Test_CV set : ',len(y))



###################################################
train_pos = '../Tweets/train_pos.txt'
train_neg = '../Tweets/train_neg.txt'
train_pos_correct = 'train_pos_correct.txt'

train_pos_processed = 'train_pos_processed.txt'
train_neg_processed = 'train_neg_processed.txt'
train_neg_correct = 'train_neg_correct.txt'

test = '../Tweets/test_data.txt'
test_processed = 'test_data_processed.txt'

# Make sure you select the correct lines


#----------------------------------------------------------------
# PREPROCESSING : should be done only once to generate the files
# ---------------------------------------------------------------
preprocess(train_pos, train_pos_processed, full=False)
preprocess(train_neg, train_neg_processed, full=False)
#preprocess(test, test_processed, remove_duplicates=False, remove_index=True)


#-------------------------------------------------------------------------------
# CROSS VALIDATION : choose the percentage of the CV
#---------------------------------------------------------------------------------
separate_crossval(train_pos_processed,'train_pos_processed_CV.txt',train_neg_processed,'train_neg_processed_CV.txt','test_CV.txt',0.05,full=False)





