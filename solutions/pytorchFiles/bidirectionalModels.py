import torch

from torch import nn

class GRU(nn.Module):

    def __init__(self, inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, numOfLayers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(inputDimensions, embeddingDimensions)
        self.rnn = nn.GRU(embeddingDimensions, hiddenDimensions, num_layers=numOfLayers, bidirectional=True, dropout=dropout)
        self.fullyConnectedLayer = nn.Linear(hiddenDimensions * 2, outputDimensions)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        return self.fullyConnectedLayer(hidden.squeeze(0))


class LSTM(nn.Module):

    def __init__(self, inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, numOfLayers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(inputDimensions, embeddingDimensions)
        self.rnn = nn.LSTM(embeddingDimensions, hiddenDimensions, num_layers=numOfLayers, bidirectional=True, dropout=dropout)
        self.fullyConnectedLayer = nn.Linear(hiddenDimensions * 2, outputDimensions)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, _) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        return self.fullyConnectedLayer(hidden.squeeze(0))


class RNN(nn.Module):

    def __init__(self, inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, numOfLayers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(inputDimensions, embeddingDimensions)
        self.rnn = nn.RNN(embeddingDimensions, hiddenDimensions, num_layers=numOfLayers, bidirectional=True, dropout=dropout)
        self.fullyConnectedLayer = nn.Linear(hiddenDimensions * 2, outputDimensions)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        return self.fullyConnectedLayer(hidden.squeeze(0))