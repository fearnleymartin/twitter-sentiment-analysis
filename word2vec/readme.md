# Tweet sentiment analysis by using Gensim's Word2Vec model

This is a demonstration case using Gensim's Word2Vec algorithm.

We chose to include this as a baseline for sentiment analyzing and to contrast with FastText's approach.

This model is running on the non-full train sets (but can be tweaked to be extended on the full sets) in order to just show a rough idea of its performances. An improvement of 3-4% can be achieved when running on full train set.

## Instructions

Run the `classifier.py` code to produce the predictions on test data. It uses the pre-computed Word2Vec model 'model.bin'. Each tweet will be translated into a vector which is the sum of the embeddings of each of its words, and then used to train a simple Multi-layer perceptron to achieve 75% accuracy.

To generate a new Word2Vec model, run the script `generate_embeddings.py`, by specifying the input and output files inside the script.

This will generate a vector for each tweet by summing their word embeddings and then run a random forest followed by cross validation.


