from nltk.tokenize.casual import TweetTokenizer
import re

full = False
tokenizer = TweetTokenizer(preserve_case=False)
dev_sample_percentage=0.1

train_pos = '../Tweets/train_pos.txt'
train_neg = '../Tweets/train_neg.txt'
train_pos_processed = 'train_pos_processed.txt'
train_neg_processed = 'train_neg_processed.txt'
test = '../Tweets/test_data.txt'
test_processed = 'test_data_processed.txt'




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

def separate_crossval(input_file_pos,output_file_pos,input_file_neg,output_file_neg,output_file_crossval,percentage)
    with open(input_file_pos, encoding='utf8') as file_pos:
        tweets_pos=file_pos.readlines()
    withopen(input_file_neg,encoding='utf8') as file_neg:
        tweets_neg=file_neg.readlines()





# Make sure you select the correct lines

#preprocess(train_pos, train_pos_processed, full=True)
#preprocess(train_neg, train_neg_processed, full=True)


dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print(x_train.shape[0])
print(x_train.shape[1])


preprocess(test, test_processed, remove_duplicates=False, remove_index=True)


