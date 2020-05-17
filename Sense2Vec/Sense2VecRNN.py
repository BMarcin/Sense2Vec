import torch
import torch.nn as nn


class Sense2VecRRN(nn.Module):
    def __init__(self, window_size, len_token2idx):
        super(Sense2VecRRN, self).__init__()

        self.window_size = window_size
        self.len_token2idx = len_token2idx
        self.hidden_size = 100

        self.rnn = nn.GRU(len_token2idx, self.hiddensize, batch_first=True, bias=False, bidirectional=True,
                          num_layers=2)

        self.lin = nn.Linear(2 * self.hiddensize, len_token2idx, bias=False)

    def init_weights(self):
        init_range = 0.1
        self.lin.weight.data.uniform_(-init_range, init_range)
        self.rnn.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, hidden):
        input_vector = torch.zeros((x.data.shape[0], x.data.shape[1], self.len_token2idx)).to(x.data.device)
        ''' rewrite to one hot vectory '''
        for c1, window in enumerate(x.data):
            for c2, value in enumerate(window):
                input_vector[c1, c2, value] = 1

        ''' pass variables through RNN '''
        input_vector, hidden = self.rnn(input_vector, hidden)
        prediction_vector = self.lin(input_vector)

        return prediction_vector, hidden
