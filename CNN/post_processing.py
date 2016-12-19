import numpy as np
from helpers_py import export_predictions

input_file = 'predictions.npy'  # file to process
output_file = 'Short_without_word2vec'  # file formatted for kaggle submission

if __name__ == "__main__":
    predictions = np.load(input_file)

    converted_pred = np.copy(predictions)

    for index in np.linspace(0, len(predictions)-1, len(predictions)):
        if predictions[index] == 0:
            converted_pred[index] = -1
        else:
            converted_pred[index] = 1
    print(predictions)
    print(converted_pred)

    export_predictions(converted_pred, output_file)