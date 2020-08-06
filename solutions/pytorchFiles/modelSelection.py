import pytorchFiles.forwardModels as forwardModels
import pytorchFiles.bidirectionalModels as bidirectionalModels

from torch.nn import DataParallel

def defineForwardModel(cell, inputDimensions, embeddingDimensions, hiddenDimensions, dropout):
    model = None
    outputDimensions = 1

    if dropout != 0:
        cell += 'Dropout'
    
    if cell == 'GRU':
        model = forwardModels.GRU(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions)
    elif cell == 'LSTM':
        model = forwardModels.LSTM(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions)
    elif cell == 'RNN':
        model = forwardModels.RNN(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions)
    elif cell == 'GRUDropout':
        model = forwardModels.GRUDropout(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, dropout)
    elif cell == 'LSTMDropout':
        model = forwardModels.LSTMDropout(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, dropout)
    elif cell == 'RNNDropout':
        model = forwardModels.RNNDropout(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, dropout)

    return model


def defineBidirectionalModel(cell, inputDimensions, embeddingDimensions, hiddenDimensions, numOfLayers, dropout):
    model = None
    outputDimensions = 1

    if cell == 'GRU':
        model = bidirectionalModels.GRU(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, numOfLayers, dropout)
    elif cell == 'LSTM':
        model = bidirectionalModels.LSTM(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, numOfLayers, dropout)
    elif cell == 'RNN':
        model = bidirectionalModels.RNN(inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, numOfLayers, dropout)

    return model