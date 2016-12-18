import re
import numpy as np

full = False



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
    predictions=[]
    with open(output_file, 'w', encoding='utf8') as of:
        of.write('Id,Prediction\n')
        with open(input_file, encoding='utf8') as f:
            for index, line in enumerate(f):
                if line == '__label__NEG,\n':
                    of.write(str(index+1)+','+'-1\n')
                    predictions.append(-1)
                elif line == '__label__POS,\n':
                    of.write(str(index+1)+','+'1\n')
                    predictions.append(1)
                else:
                    raise Exception('Unknown label', line)
    np.save('predictions',predictions)



def accuracy(y_CV, predictions):
    y_CV=np.load(y_CV)
    predictions=np.load(predictions)
    if not len(y_CV) == len(predictions):
        raise Exception('Vectors should be the same size!')
    print('Proportion of positive in predictions:', np.sum(predictions==1)/len(predictions))
    print('Proportion of positive in labels:', np.sum(y_CV==1)/len(y_CV))
    accuracy=np.sum(y_CV==predictions)/len(y_CV)
    return accuracy


####################################################################################
train_pos_processed = 'train_pos_processed.txt'
train_neg_processed = 'train_neg_processed.txt'
train_fasttext = 'train_fasttext.txt'
train_fasttext_no_label = 'train_fasttext_no_label.txt'
train_pos_processed_CV = 'train_pos_processed_CV.txt'
train_neg_processed_CV = 'train_neg_processed_CV.txt'
# train_pos_fasttext = 'train_pos_processed_fasttext.txt'
# train_neg_fasttext = 'train_neg_processed_fasttext.txt'

# Make sure to select correct lines to run

#------------------------------------------------------
# TO OBTAIN WORD REPRESENTATION
#------------------------------------------------------
#preprocess(train_pos_processed, train_neg_processed, train_fasttext_no_label, label=False, full=True)

#------------------------------------------------------
# GENERATE THE PROPER TRAIN DATASET FOR FASTTEXT :
# To be done only once for a given train dataset
#------------------------------------------------------

#preprocess(train_pos_processed_CV, train_neg_processed_CV, train_fasttext, full=True)


#--------------------------------------------------------
# POSTPROCESSING
# Generate the output file and computes the accuracy in the cross validation
#--------------------------------------------------------

postprocess('results_fasttext.txt', 'results_fasttext_processed.txt', full=True)
print('The accuracy of this method is :', accuracy('y_CV.npy','predictions.npy'))