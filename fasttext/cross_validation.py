import numpy as np
import os

#-------------------------------------------------------------------------------
# CROSS VALIDATION FOR FASTTEXT :
#---------------------------------------------------------------------------------

dir_path = '../../../vagrant/fasttext/'
input_tweets_file = dir_path + '../data/fasttext/train_fasttext.txt'

def build_k_indices(input_tweets, k_fold, seed):
    """build k indices for k-fold."""
    num_row = len(input_tweets)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(input_tweets, k_indices, k, remove_files=True, num_epochs=5, wordNgrams=2, dim=100):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    input_tweets = np.array(input_tweets)
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = input_tweets[te_indice]
    y_tr = input_tweets[tr_indice]

    train_path = dir_path + 'cross_validation_data/train'+str(k)+'.txt'
    test_path = dir_path + 'cross_validation_data/test_data'+str(k)+'.txt'
    test_labels_path = dir_path + 'cross_validation_data/test_label' + str(k) + '.txt'
    model_path = dir_path + 'cross_validation_data/train_model_' + str(k)
    test_predictions_path = dir_path + 'cross_validation_data/test_output'+str(k)+'.txt'

    with open(train_path, 'w') as f:
        for tweet in y_tr:
            f.write(tweet)
    with open(test_path, 'w') as test_file:
        with open(test_labels_path, 'w') as label_file:
            for tweet in y_te:
                label, tweet = tweet.split(',')[0]+',', tweet[len(tweet.split(',')[0])+2:]
                test_file.write(tweet)
                label_file.write(label+'\n')

    train_command = './fasttext supervised -input {} -output {} -wordNgrams {} -epoch {} -dim {}'.format(train_path, model_path, wordNgrams, num_epochs, dim)

    eval_command = './fasttext predict {}.bin {} > {}'.format(model_path, test_path, test_predictions_path)

    os.system(train_command)
    os.system(eval_command)

    with open(test_predictions_path) as f:
        predictions = f.readlines()
    with open(test_labels_path) as f:
        correct_labels = f.readlines()

    total = 0
    for label1, label2 in zip(predictions, correct_labels):
        if label1 == label2:
            total += 1
    accuracy = float(total) / float(len(predictions))
    print('accuracy for fold {} is {}'.format(k, accuracy))

    if remove_files:
        os.system('rm -r ' + dir_path + 'cross_validation_data/*')
    return accuracy

if __name__ == "__main__":
    with open(input_tweets_file) as f:
        input_tweets = f.readlines()
    k_fold = 3
    k_indices = build_k_indices(input_tweets, k_fold, seed=3)
    scores = []
    for k in range(k_fold):
        print('fold: ', k)
        scores.append(cross_validation(input_tweets, k_indices, k))
    print(scores)
    print('average score: ', sum(scores)/len(scores))
