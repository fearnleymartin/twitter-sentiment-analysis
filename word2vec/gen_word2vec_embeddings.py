from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('train_full_cleaned.txt')

model = word2vec.Word2Vec(sentences, size=50, min_count=5)

model.save_word2vec_format('word2vec.model.bin', binary=True)

