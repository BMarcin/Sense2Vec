import os
import random
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.optim as optim
from poutyne.framework import Experiment
from torch.utils.data import DataLoader

from Sense2Vec.DS import DS
from Sense2Vec.Sense2VecCBOW import Sense2VecCBOW

torch.manual_seed(1010101011)
random.seed(1010101011)

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
        "--experiment_path",
        dest="experiment_path",
        help="Dir path to save experiment",
        metavar="PATH"
    )

    parser.add_option(
        "--embeddings_size",
        dest="embeddings_size",
        help="Embedding vector size",
        type=int
    )

    parser.add_option(
        "--target_vectors",
        dest="target_vectors",
        help="Target vector size for each token",
        type=int
    )

    parser.add_option(
        "--minimal_token_occurences",
        dest="minimal_token_occurences",
        type=int
    )

    options, args = parser.parse_args()

    experiment_path = options.experiment_path
    lr = options.lr
    bs = options.bs
    seq_len = options.seq_len
    epochs = options.epochs
    device = torch.device(options.device)
    minimal_token_occurences = options.minimal_token_occurences
    dataset_pickle_path = options.dataset_pickle_path

    assert seq_len % 2 == 1, 'Seq len has to be odd number'

    token2idx_save_path = os.path.join(dataset_pickle_path,
                                       "ds_t2x_s{}_c{}.pth".format(str(seq_len), str(minimal_token_occurences)))
    dataset_x_save_path = os.path.join(dataset_pickle_path,
                                       "ds_x_s{}_c{}.pth".format(str(seq_len), str(minimal_token_occurences)))
    dataset_y_save_path = os.path.join(dataset_pickle_path,
                                       "ds_y_s{}_c{}.pth".format(str(seq_len), str(minimal_token_occurences)))

    if os.path.exists(token2idx_save_path) and os.path.exists(dataset_x_save_path) and os.path.exists(
            dataset_y_save_path):
        print("Dataset exists")
        ds = DS(
            options.input_corpus,
            options.seq_len,
            dataset_x=torch.load(dataset_x_save_path),
            dataset_y=torch.load(dataset_y_save_path),
            token2idx=torch.load(token2idx_save_path),
            minimal_word_occurences=minimal_token_occurences
        )
    else:
        ds = DS(options.input_corpus, options.seq_len, minimal_word_occurences=minimal_token_occurences)
        torch.save(ds.token2idx, token2idx_save_path)
        torch.save(ds.ds_x, dataset_x_save_path)
        torch.save(ds.ds_y, dataset_y_save_path)

    print("DS unique values", len(ds.token2idx))

    DL = DataLoader(dataset=ds, batch_size=bs, num_workers=4, shuffle=True)

    model = Sense2VecCBOW(
        len(ds.token2idx),
        options.embeddings_size,
        options.target_vectors,
        options.seq_len
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    experiment = Experiment(
        experiment_path,
        model,
        optimizer=optimizer,
        loss_function=criterion,
        batch_metrics=['accuracy'],
        monitor_metric='acc',
        monitor_mode='max',
        device=device
    )

    experiment.train(DL, epochs=epochs)
