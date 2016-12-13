import numpy as np
from helpers_py import export_predictions


predictions = np.load('predictions.npy')
print(len(predictions))
converted_pred = np.copy(predictions)


print(predictions)
print(converted_pred)

for index in np.linspace(0,len(predictions)-1,len(predictions)):
    if predictions[index] == 0:
        converted_pred[index]= -1
    else:
        converted_pred[index]= 1


export_predictions(converted_pred,'test_cnn_5')