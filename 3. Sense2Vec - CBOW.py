import os
import random
import string
from optparse import OptionParser

import torch
import torch.nn as nn
from poutyne.framework import Model, Experiment
from torch import optim
from torch.utils.data import DataLoader

from Sense2Vec.DS2 import DS
from Sense2Vec.Sense2VecCBOW import Sense2VecCBOW

torch.manual_seed(1010101011)
random.seed(1010101011)


def get_experiment_id(path):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(32))

    while os.path.exists(os.path.join(path, result_str)):
        result_str = ''.join(random.choice(letters) for i in range(32))

    return result_str


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
        "--experiment",
        dest="experiment",
        type=str
    )

    parser.add_option(
        "--minimal_token_occurences",
        dest="minimal_token_occurences",
        type=int
    )

    options, args = parser.parse_args()

    experiment_id = options.experiment

    lr = options.lr
    bs = options.bs
    seq_len = options.seq_len
    epochs = options.epochs
    device = torch.device(options.device)
    minimal_token_occurences = options.minimal_token_occurences

    assert seq_len % 2 == 1, 'Seq len has to be odd number'

    if os.path.exists(os.path.join(options.dataset_pickle_path,
                                   "ds_token2idx__seq_len_{}__min_token_occ_{}.pth".format(
                                       str(seq_len), str(minimal_token_occurences)))) and os.path.exists(
        os.path.join(options.dataset_pickle_path,
                     "ds_dataset__seq_len_{}__min_token_occ_{}.pth".format(str(seq_len),
                                                                           str(minimal_token_occurences)))):
        print("Dataset exists")
        ds = DS(
            options.input_corpus,
            options.seq_len,
            dataset=torch.load(
                os.path.join(options.dataset_pickle_path, "ds_dataset__seq_len_{}__min_token_occ_{}.pth".format(
                    str(seq_len), int(minimal_token_occurences)))),
            token2idx=torch.load(
                os.path.join(options.dataset_pickle_path, "ds_token2idx__seq_len_{}__min_token_occ_{}.pth".format(
                    str(seq_len), int(minimal_token_occurences)))),
            minimal_word_occurences=minimal_token_occurences
        )
    else:
        ds = DS(options.input_corpus, options.seq_len, minimal_word_occurences=minimal_token_occurences)
        torch.save(ds.token2idx,
                   os.path.join(options.dataset_pickle_path, "ds_token2idx__seq_len_{}__min_token_occ_{}.pth".format(
                       str(seq_len), int(minimal_token_occurences))))
        torch.save(ds.dataset,
                  os.path.join(options.dataset_pickle_path, "ds_dataset__seq_len_{}__min_token_occ_{}.pth".format(
                       str(seq_len), int(minimal_token_occurences))))

        print("DS unique values", len(ds.token2idx))

    DL = DataLoader(dataset=ds, batch_size=bs, num_workers=4, shuffle=True)
    # os.mkdir(os.path.join(options.model_pickles_dir_path, experiment_id))

    model = Sense2VecCBOW(
        len(ds.token2idx),
        options.embeddings_size,
        options.target_vectors,
        options.seq_len
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    experiment = Experiment(
        'experiments/'+experiment_id,
        model,
        optimizer=optimizer,
        loss_function=criterion,
        batch_metrics=['accuracy'],
        monitor_metric='acc',
        device=device
    )

    experiment.train(DL, epochs=epochs)