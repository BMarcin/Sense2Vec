import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json

from Sense2Vec.DS2 import DS2 as DS
from Sense2Vec.Sense2VecRNN import Sense2VecRRN


def train(epochs, criterion, optimizer, model, dataloader, savepath):
    # t_epochs = tqdm(range(epochs))

    for epoch in range(epochs):
        t_batch = tqdm(dataloader, leave=False)

        epoch_loss = []

        hidden = torch.zeros((2 * 2, dataloader.batch_size, 100)).to(device)
        for batch in t_batch:
            # lenghts = batch[0].to(device)
            x = batch.text
            y = batch.target

            model.train()
            optimizer.zero_grad()

            y_, hidden = model(x, hidden)
            hidden = hidden.detach()

            loss = criterion(y_.transpose(1, 2), y.data)
            loss.backward()
            optimizer.step()

            losss = loss.item()
            epoch_loss.append(losss)

            t_batch.set_description("Loss: {:.8f}".format(np.mean(epoch_loss[-1000:])))
        t_batch.close()
        # t_epochs.set_description("Epoch {}/{}".format(epoch + 1, epochs))

        torch.save(model, os.path.join(savepath, "model_bidirectional_{}.pth".format(epoch + 1)))
        print("Epoch {}/{}, Loss {:.8f}".format(epoch + 1, epochs, losss))


def save_weights(output_file_path, dataset, model):
    sense2vec = {}

    for token, weight in tqdm(zip(dataset.token2idx.keys(), model.lin.weight.cpu().detach().tolist()),
                                       total=len(ds.token2idx.keys())):
        sense2vec[token] = weight

    with open(output_file_path, 'w', encoding="utf8") as f:
        json.dump(sense2vec, f, indent=1)


if __name__ == '__main__':
    lr = 3e-3
    bs = 32
    seq_len = 10
    epochs = 3
    device = torch.device("cuda")

    ds = DS(
        os.path.join("data", "tagged_data_big", "corpus_combined.txt"),
        bs,
        seq_len,
        device
    )
    DL = ds.build_iterator()
    model = Sense2VecRRN(ds.bptt_len, len(ds.TEXT.vocab)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(
        epochs,
        criterion,
        optimizer,
        model,
        DL,
        os.path.join('data', 'models')
    )

    # save_weights(
    #     os.path.join("data", "results", "sense2vec_big.json"),
    #     ds,
    #     model
    # )
