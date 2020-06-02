import pytorchFiles.models as models

from torch.nn import DataParallel

def defineModel(rnnClass, inputDimensions, embeddingDimensions, hiddenDimensions, dropout):
    model = None
    outputDimensions = 1

    if rnnClass == 'RNN':
        model = models.RNN(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions)
    elif rnnClass == 'RNNDropout':
        model = models.RNNDropout(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, dropout)
    elif rnnClass == 'LSTM':
        model = models.LSTM(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions)
    elif rnnClass == 'LSTMDropout':
        model = models.LSTMDropout(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, dropout)

    model = DataParallel(model, device_ids=[0])
    return model