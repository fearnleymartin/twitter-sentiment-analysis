"""
To run FastText, we need to combine our positive and negative tweets into a single data file
where each tweet in preceded by its label, for ex '__label__POS, tweet'
"""


def fasttext_formatting(input_file_pos, input_file_neg, output_file, label=True):
    """
    Formats tweets to use with fasttext, add label at beginning of each tweet
    :param input_file_pos:
    :param input_file_neg:
    :param output_file: combines pos and neg to a single file
    :param full: choose full data set or not
    :param label: for test data set, we don't want to add a label
    :return:
    """
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


####################################################################################
# INPUT/OUTPUT FILE PARAMS
# Input
train_pos_processed = '../data/processed/train_pos_processed_full.txt'
train_neg_processed = '../data/processed/train_neg_processed_full.txt'

# Output
train_fasttext = '../data/fasttext/train_fasttext_full.txt'

#------------------------------------------------------
# GENERATE THE CORRECT FORMATTING FOR THE TRAIN DATASET FOR FASTTEXT :
# To be done only once for a given train dataset
#------------------------------------------------------

if __name__ == "__main__":
    fasttext_formatting(train_pos_processed, train_neg_processed, train_fasttext)

