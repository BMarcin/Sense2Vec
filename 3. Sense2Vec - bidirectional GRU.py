import os
from optparse import OptionParser

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import json

from Sense2Vec.DS2 import DS2 as DS
from Sense2Vec.Sense2VecRNN import Sense2VecRRN


def train(epochs, criterion, optimizer, model, dataloader, savepath):
    for epoch in range(epochs):
        t_batch = tqdm(dataloader, leave=False)

        epoch_loss = []

        hidden = torch.zeros((2 * 2, dataloader.batch_size, 100)).to(device)
        for batch in t_batch:
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

        torch.save(model, os.path.join(savepath, "model_bidirectional_{}.pth".format(epoch + 1)))
        print("Epoch {}/{}, Loss {:.8f}".format(epoch + 1, epochs, losss))


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option(
        "--lr",
        dest="lr",
        help="learning rate value",
        type=float,
        default=3e-3
    )

    parser.add_option(
        "--bs",
        dest="bs",
        help="Batch size",
        metavar="INT",
        type=int,
        default=28
    )

    parser.add_option(
        "--seq_len",
        dest="seq_len",
        help="Sequence length",
        metavar="INT",
        type=int,
        default=10
    )

    parser.add_option(
        "--epochs",
        dest="epochs",
        help="Epochs number",
        metavar="INT",
        type=int,
        default=1
    )

    parser.add_option(
        "--device",
        dest="device",
        help="Device, for example: cuda, cuda:1, cpu",
        type=str,
        default="cuda"
    )

    parser.add_option(
        "--input_corpus",
        dest="input_corpus",
        help="Corpus file from 2. file",
        metavar="FILE",
    )

    parser.add_option(
        "--dataset_pickle_path",
        dest="dataset_pickle_path",
        help="Path to save pickled dataset",
        metavar="FILE"
    )

    parser.add_option(
        "--model_pickles_dir_path",
        dest="model_pickles_dir_path",
        help="Dir path to save model each epoch",
        metavar="PATH"
    )

    options, args = parser.parse_args()

    lr = options.lr
    bs = options.bs
    seq_len = options.seq_len
    epochs = options.epochs
    device = torch.device(options.device)

    ds = DS(
        options.input_corpus,
        bs,
        seq_len,
        device
    )
    torch.save(ds, options.dataset_pickle_path)

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
        options.model_pickles_dir_path
    )
