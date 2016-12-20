**[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## Training

Train:

```bash
./train.py
```

You can specify the input/outputs with the flags in train.py.

We have made it possible ti initialise the cnn with the word embeddings from both word2vec and fasttext.

It is also possible to modify the model hyperparameters straight from the flags in train.py

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.

## Post-processing
Specify input/output files and run cnn/postprocessing.py

This will produce a kaggle comopatible submission file from the output given by the cnn.

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)