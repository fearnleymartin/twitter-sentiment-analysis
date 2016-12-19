"""
We use gensim's implementation of word2vec to generate vector representation of all the words in our tweets
"""

from gensim.models import word2vec
import logging
import re

train_pos = 'train_pos_processed_full.txt'
train_neg = 'train_neg_processed_full.txt'

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
    sentences = MySentences(train_pos, train_neg)
    model = word2vec.Word2Vec(sentences, size=50, min_count=3)
    model.save_word2vec_format('word2vec.preprocessed.full.model.bin', binary=True)
    print('created word2vec model')
