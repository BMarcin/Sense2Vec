import torch
import torch.nn as nn

class Sense2VecRRN(nn.Module):
    def __init__(self, windowsize, len_token2idx):
        super(Sense2VecRRN, self).__init__()

        self.windowsize = windowsize
        self.len_token2idx = len_token2idx
        self.hiddensize = 100

        self.rnn = nn.GRU(len_token2idx, self.hiddensize, batch_first=True, bias=False, bidirectional=True,
                          num_layers=2)

        self.lin = nn.Linear(2 * self.hiddensize, len_token2idx, bias=False)

    def init_weights(self):
        initrange = 0.1
        self.lin.weight.data.uniform_(-initrange, initrange)
        self.rnn.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        #         print(x.data.shape)

        inputvector = torch.zeros((x.data.shape[0], x.data.shape[1], self.len_token2idx)).to(x.data.device)
        ''' przepisanie na one hot vectory '''
        for c1, window in enumerate(x.data):
            for c2, value in enumerate(window):
                inputvector[c1, c2, value] = 1

        # hidden = torch.zeros((2 * 2, x.data.shape[0], self.hiddensize)).to(x.data.device)

        ''' przepuszczenie przez RNN stacka '''
        inputvector, hidden = self.rnn(inputvector, hidden)
        #         print(inputvector.shape)
        predictionvector = self.lin(inputvector)

        return predictionvector, hidden