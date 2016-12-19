from gensim.models import word2vec
import logging
import re

train_pos = 'train_pos.txt'
train_neg = 'train_neg.txt'
train_pos_processed = 'train_pos_processed_full.txt'
train_neg_processed = 'train_neg_processed_full.txt'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
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

sentences = MySentences(train_pos_processed, train_pos_processed)

model = word2vec.Word2Vec(sentences, size=50, min_count=3)

model.save_word2vec_format('word2vec.preprocessed.full.model.bin', binary=True)
print('created word2vec model')

def test_word2vec(model_file):
    model = word2vec.Word2Vec.load_word2vec_format(model_file, binary=True)
    print(model.most_similar('good'))
    print(model.most_similar('excellent'))
    print(model.most_similar('boy'))
    print(model.most_similar('phone'))

# test_word2vec('word2vec.preprocessed.full.model.bin')