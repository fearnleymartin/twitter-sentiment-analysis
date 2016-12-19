# Generating word embeddings

Run in order (and specify correct input/output paths):

```
build_vocab.sh
cut_vocab.sh
python pickle_vocab.py
python glove.py
```

This generates vector embeddings for each word in the tweets which are stored in a numpy array

# Predictions

Specify correct input/outputs and run

```
python classifier.py
```

This will a vector for each tweet by summing their word embeddings and then run a random forest followed by cross validation.
