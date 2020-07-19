import os
from optparse import OptionParser

import mlflow
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from Sense2Vec.DS import DS
from Sense2Vec.Sense2VecCBOW import Sense2VecCBOW


def train(epochs, criterion, optimizer, model, dataloader, savepath, device, save_each=None):
    for epoch in range(epochs):
        t_batch = tqdm(dataloader, leave=False)

        epoch_loss = []

        for i, batch in enumerate(t_batch):
            x = batch[0].long().to(device)
            y = batch[1].long().to(device)

            model.train()
            optimizer.zero_grad()

            y_ = model(x)
            single_loss = criterion(y_, y)
            single_loss.backward()
            optimizer.step()

            single_loss_value = single_loss.item()
            epoch_loss.append(single_loss_value)

            t_batch.set_description("Loss: {:.8f}".format(np.mean(epoch_loss[-1000:])))

            mlflow.log_metric('bs_loss_mean', np.mean(epoch_loss[-1000:]), step=epoch + 1)

            if save_each:
                if i % save_each == 0 and i != 0:
                    torch.save(model.state_dict(),
                               os.path.join(savepath, mlflow.active_run().info.run_id,
                                            "model_cbow_step_{}_epoch_{}.pth".format(i, epoch + 1)))
        t_batch.close()
        mlflow.log_metric('epoch_loss_mean', np.mean(epoch_loss), step=epoch + 1)

        torch.save(model.state_dict(),
                   os.path.join(savepath, mlflow.active_run().info.run_id, "model_cbow_epoch{}.pth".format(epoch + 1)))
        print("Epoch {}/{}, Loss {:.8f}".format(epoch + 1, epochs, single_loss_value))

    mlflow.log_artifact(os.path.abspath(os.path.join(savepath, mlflow.active_run().info.run_id)))


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
        "--mlflow_host",
        dest="mlflow_host",
        type=str
    )

    parser.add_option(
        "--mlflow_experiment",
        dest="mlflow_experiment",
        type=str
    )

    parser.add_option(
        "--minimal_token_occurences",
        dest="minimal_token_occurences",
        type=int
    )

    options, args = parser.parse_args()

    mlflow.set_tracking_uri(options.mlflow_host)
    mlflow.set_experiment(options.mlflow_experiment)

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
                     "ds_tokens__seq_len_{}__min_token_occ_{}.pth".format(str(seq_len),
                                                                          str(minimal_token_occurences)))):
        print("Dataset exists")
        ds = DS(
            options.input_corpus,
            options.seq_len,
            tokens=torch.load(
                os.path.join(options.dataset_pickle_path, "ds_tokens__seq_len_{}__min_token_occ_{}.pth".format(
                    str(seq_len), str(minimal_token_occurences)))),
            token2idx=torch.load(
                os.path.join(options.dataset_pickle_path, "ds_token2idx__seq_len_{}__min_token_occ_{}.pth".format(
                    str(seq_len), str(minimal_token_occurences))))
        )
    else:
        ds = DS(options.input_corpus, options.seq_len, minimal_token_occurences)
        torch.save(ds.token2idx,
                   os.path.join(options.dataset_pickle_path, "ds_token2idx__seq_len_{}__min_token_occ_{}.pth".format(
                       str(seq_len, minimal_token_occurences))))
        torch.save(ds.tokens,
                   os.path.join(options.dataset_pickle_path, "ds_tokens__seq_len_{}__min_token_occ_{}.pth".format(
                       str(seq_len, minimal_token_occurences))))

    with mlflow.start_run():
        mlflow.log_param('lr', lr)
        mlflow.log_param('bs', bs)
        mlflow.log_param('seq_len', seq_len)
        mlflow.log_param('epochs', epochs)

        print("DS unique values", len(ds.token2idx))

        DL = DataLoader(dataset=ds, batch_size=bs, num_workers=6)
        os.mkdir(os.path.join(options.model_pickles_dir_path, mlflow.active_run().info.run_id))

        model = Sense2VecCBOW(
            len(ds.token2idx),
            options.embeddings_size,
            options.target_vectors,
            options.seq_len
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train(
            epochs,
            criterion,
            optimizer,
            model,
            DL,
            options.model_pickles_dir_path,
            device,
            save_each=40000
        )
