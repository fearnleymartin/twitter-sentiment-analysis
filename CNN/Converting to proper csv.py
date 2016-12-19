import numpy as np
from cnn.helpers_py import export_predictions


predictions = np.load('predictions.npy')

converted_pred = np.copy(predictions)

for index in np.linspace(0, len(predictions)-1, len(predictions)):
    if predictions[index] == 0:
        converted_pred[index] = -1
    else:
        converted_pred[index] = 1
print(predictions)
print(converted_pred)

export_predictions(converted_pred,'Short_without_word2vec')