"""
We need to convert the output files produced by fasttest into a file format accepted by kaggle
"""
import re
import numpy as np


def fasttext_postprocess(input_file, output_file, full=False):
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
    predictions = []
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
    np.save('predictions', predictions)


def accuracy(y_CV, predictions):
    y_CV = np.load(y_CV)
    predictions = np.load(predictions)
    if not len(y_CV) == len(predictions):
        raise Exception('Vectors should be the same size!')
    print('Proportion of positive in predictions:', np.sum(predictions == 1)/len(predictions))
    print('Proportion of positive in labels:', np.sum(y_CV == 1)/len(y_CV))
    accuracy = np.sum(y_CV == predictions)/len(y_CV)
    return accuracy


####################################################################################
# Input
results_file = 'results_fasttext.txt'

# Output
results_file_processed = 'results_fasttext_processed.txt'

#--------------------------------------------------------
# POSTPROCESSING
# Generate the output file and computes the accuracy in the cross validation
#--------------------------------------------------------

if __name__ == "__main__":
    fasttext_postprocess(results_file, results_file_processed, full=False)
    print('The accuracy of this method is :', accuracy('y_CV.npy','predictions.npy'))