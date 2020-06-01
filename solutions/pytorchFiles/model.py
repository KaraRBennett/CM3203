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
        output, hidden = self.rnn(embedded)
        hidden1D = hidden.squeeze(0)

        assert torch.equal(output[-1, :, :], hidden1D)

        return self.fullyConnectedLayer(hidden1D)