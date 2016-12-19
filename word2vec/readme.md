# Generating the word embeddings

Run word2vec/generate_embeddings.py after specifying input/output files

This will generate a binary file storing all the word representation as learnt from the input tweets

# Predictions

Run word2vec/classifier.py after specifying input/output files

This will a vector for each tweet by summing their word embeddings and then run a random forest followed by cross validation.

