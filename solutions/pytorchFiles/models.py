import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions):
        super().__init__()

        self.embedding = nn.Embedding(inputDimensions, embeddingDimensions)
        self.rnn = nn.RNN(embeddingDimensions, hiddenDimensions)
        self.fullyConnectedLayer = nn.Linear(hiddenDimensions, outputDimensions)
    

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden) = self.rnn(embedded)
        hidden1D = hidden.squeeze(0)

        assert torch.equal(output[-1, :, :], hidden1D)

        return self.fullyConnectedLayer(hidden1D)


class RNNDropout(nn.Module):

    def __init__(self, inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, dropout):
        super().__init__()

        self.embedding = nn.Embedding(inputDimensions, embeddingDimensions)
        self.rnn = nn.RNN(embeddingDimensions, hiddenDimensions)
        self.fullyConnectedLayer = nn.Linear(hiddenDimensions, outputDimensions)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, text):
        embedded = self.embedding(text)
        embeddedDropout = self.dropout(embedded)
        output, (hidden) = self.rnn(embeddedDropout)
        hidden1D = hidden.squeeze(0)

        assert torch.equal(output[-1, :, :], hidden1D)

        return self.fullyConnectedLayer(hidden1D)


class LSTM(nn.Module):

    def __init__(self, inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions):
        super().__init__()

        self.embedding = nn.Embedding(inputDimensions, embeddingDimensions)
        self.rnn = nn.LSTM(embeddingDimensions, hiddenDimensions)
        self.fullyConnectedLayer = nn.Linear(hiddenDimensions, outputDimensions)
    

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.rnn(embedded)
        hidden1D = hidden.squeeze(0)

        assert torch.equal(output[-1, :, :], hidden1D)

        return self.fullyConnectedLayer(hidden1D)


class LSTMDropout(nn.Module):

    def __init__(self, inputDimensions, embeddingDimensions, hiddenDimensions, outputDimensions, dropout):
        super().__init__()

        self.embedding = nn.Embedding(inputDimensions, embeddingDimensions)
        self.rnn = nn.LSTM(embeddingDimensions, hiddenDimensions)
        self.fullyConnectedLayer = nn.Linear(hiddenDimensions, outputDimensions)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, text):
        embedded = self.embedding(text)
        embeddedDropout = self.dropout(embedded)
        output, (hidden, _) = self.rnn(embeddedDropout)
        hidden1D = hidden.squeeze(0)

        assert torch.equal(output[-1, :, :], hidden1D)

        return self.fullyConnectedLayer(hidden1D)