import torch
import torch.nn as nn
import torch.functional as F


class Sense2VecCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, vectors, sequence_length):
        super(Sense2VecCBOW, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.vectors = vectors

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.fc_in = nn.Linear(embedding_size * (sequence_length - 1), vectors)
        self.fc_out = nn.Linear(vectors, vocab_size)

    def init_weights(self):
        init_range = 0.1
        self.lin.weight.data.uniform_(-init_range, init_range)
        self.rnn.weight.data.uniform_(-init_range, init_range)

    def get_weights(self):
        return self.fc_out.weight.cpu().detach().tolist()

    def forward(self, x):
        x = self.embeddings(x)
        x = self.fc_in(x.reshape(len(x), -1))
        x = torch.relu(x)
        x = self.fc_out(x)
        return x
