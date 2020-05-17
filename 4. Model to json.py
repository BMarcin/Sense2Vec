import os
import json
import torch
import torch.nn as nn
from torch import optim
from Sense2Vec.DS2 import DS2 as DS
import tqdm
from Sense2Vec.Sense2VecRNN import Sense2VecRRN

device = torch.device("cpu")
bs = 32
seq_len = 10

ds = DS(
    os.path.join("data", "tagged_data_big", "corpus_combined.txt"),
    bs,
    seq_len,
    device
)
DL = ds.build_iterator()

model = Sense2VecRRN(ds.bptt_len, len(ds.TEXT.vocab)).to(device)
model = torch.load(os.path.join("data", "models", "model_bidirectional_2.pth"), map_location=device)

weights = model.lin.weight.detach().tolist()

sense2vec = {}

# print(ds.TEXT.vocab.stoi.keys())
keys = ds.TEXT.vocab.stoi.keys()

for token, weight in zip(keys, weights):
    sense2vec[token] = weight
#
with open(os.path.join("data", "results", "bidir.json"), 'w', encoding="utf8") as f:
    json.dump(sense2vec, f, indent=1)
