"""
We need to convert the output files produced by FastText into a file format accepted by Kaggle
"""

def fasttext_postprocess(input_file, output_file):
    """
    Converts from fasttext output format to kaggle submission format
    :param input_file:
    :param output_file:
    :return:
    """
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


####################################################################################
# Input
results_file = 'results/results.txt'

# Output
results_file_processed = 'results/results_processed.txt'

#--------------------------------------------------------
# POSTPROCESSING
# Generate the output file and computes the accuracy in the cross validation
#--------------------------------------------------------

if __name__ == "__main__":
    fasttext_postprocess(results_file, results_file_processed)
