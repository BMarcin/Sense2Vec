import torch
import torch.nn as nn
import torch.functional as F


class Sense2VecCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, vectors, sequence_length):
        super(Sense2VecCBOW, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.vectors = vectors

        # self.embeddings = nn.Embedding(vocab_size, embedding_size)
        # self.fc_in = nn.Linear(embedding_size * (sequence_length - 1), vectors)
        self.fc_in = nn.Linear(vocab_size * (sequence_length - 1), vectors)
        self.fc_out = nn.Linear(vectors, vocab_size)
        # self.pooling = nn.AvgPool1d(embedding_size, stride=2)
        # self.activation = nn.ReLU()
        # self.drop = nn.Dropout(p=0.3)
        # self.norm = nn.BatchNorm1d(vectors)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.fc_in.weight.data.uniform_(-init_range, init_range)
        self.fc_out.weight.data.uniform_(-init_range, init_range)

    def get_weights(self):
        return self.fc_out.weight.cpu().detach().tolist()

    def forward(self, x: torch.Tensor):
        y = torch.zeros((x.shape[0], x.shape[1], self.vocab_size)).to(x.device)
        y.scatter(2, x.unsqueeze(2),
                    torch.ones(x.shape[0], self.vocab_size, 1).to(x.device)
                  )
        print(x, y)
        # for batch_index, batch_unit in enumerate(x):
        #     for token_index, token in enumerate(batch_unit):
        #         y[]
        # x = self.embeddings(x)
        # print(x.shape)
        # x = x.mean(dim=1)
        # print(x.shape)
        # print(x.shape)
        # x = self.pooling(x)
        # print(x.shape)
        x = self.fc_in(y.reshape(len(x), -1))
        # x = self.fc_in(x)
        # x = torch.relu(x)
        # x = self.activation(x)
        # x = self.drop(x)
        # x = self.norm(x)
        x = self.fc_out(x)
        return x
