"""
We use gensim's implementation of word2vec to generate vector representation of all the words in our tweets
"""

from gensim.models import word2vec
import logging
import re

# Input file paths (pos and neg tweets)
train_pos = '../data/processed/train_pos_processed.txt'
train_neg = '../data/processed/train_neg_processed.txt'

# Output file path (model for word2vec)
output_model = 'model.bin'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    """
    Class for loading the positive and negative files into the word2vec class
    """
    def __init__(self, train_pos, train_neg):
        self.train_pos = train_pos
        self.train_neg = train_neg

    def __iter__(self):
        for line in open(train_pos, encoding='utf8'):
            line = re.sub(r" \n", "", line)
            yield line.split()
        for line in open(train_neg, encoding='utf8'):
            line = re.sub(r" \n", "", line)
            yield line.split()

if __name__ == "__main__":
    print('Creating Word2Vec Model')
    sentences = MySentences(train_pos, train_neg)
    model = word2vec.Word2Vec(sentences, size=50, min_count=3)
    model.save_word2vec_format(output_model, binary=True)
