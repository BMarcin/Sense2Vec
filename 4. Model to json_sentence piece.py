import os
import json
from optparse import OptionParser

from tqdm import tqdm

from Sense2Vec.Sense2VecCBOW import Sense2VecCBOW
from poutyne.framework import Experiment

import torch.nn as nn
import torch.optim as optim

from Sense2Vec.DS_padded_corpus import DS

import torch
import sentencepiece as spm
from glob import glob
import numpy as np

if __name__ == '__main__':
    parser = OptionParser()

    ds = DS(
        "data/postprocessed/wiki_sp_3000.txt",
        19,
        dataset=torch.load("data/datasets/ds_s19_c3.pth"),
        token2idx=torch.load("data/datasets/ds_t2x_s19_c3.pth"),
        minimal_word_occurences=3
    )

    model = Sense2VecCBOW(
        9151,
        100,
        200,
        19
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    experiment = Experiment(
        "experiments/wiki_sentencepiece_onehot",
        model,
        optimizer=optimizer,
        loss_function=criterion,
        batch_metrics=['accuracy'],
        monitor_metric='acc',
        monitor_mode='max',
        device="cuda:0"
    )
    experiment.load_checkpoint("best")

    weights = experiment.model.get_weights()['fc_out.weight'].cpu().detach().tolist()

    sense2vec = {}

    for token, weight in tqdm(zip(list(ds.token2idx.keys()), weights), total=len(weights)):
        sense2vec[token] = weight

    sp = spm.SentencePieceProcessor()
    sp.load('m.model')


    s2vec = {}

    with open('ds_tokens.txt', 'r', encoding='utf8') as fp:
        with open('ds_poss.txt', 'r', encoding="utf8") as fe:
            progress = tqdm()
            for sentence, poss in zip(fp, fe):
                tokens = sentence.replace("\n", "").split("\t")
                pos = poss.replace("\n", "").split("\t")

                for token, single_pos in zip(tokens, pos):
                    tokenized = sp.encode_as_pieces(token)

                    if token + "|" + single_pos.lower() not in s2vec.keys():
                        vectors = []
                        try:
                            for token_part in tokenized[1:-1]:
                                vectors.append(sense2vec[token_part + "|" + single_pos.lower()])

                            vectors = np.array(vectors)
                            vector = vectors.mean(axis=0)

                            s2vec[token + "|" + single_pos.lower()] = vector.tolist()
                            progress.update(1)
                        except Exception:
                            # print(token, tokenized, single_pos)
                            pass

    with open("data/results/wiki_sp_3000_mean.json", 'w', encoding="utf8") as f:
        json.dump(s2vec, f, indent=1)
