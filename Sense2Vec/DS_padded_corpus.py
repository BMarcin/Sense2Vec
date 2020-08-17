from collections import Counter

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np


class DS(Dataset):
    def __init__(self,
                 corpus_path,
                 window_size,
                 dataset=None,
                 token2idx=None,
                 minimal_word_occurences=3):
        self.corpus_path = corpus_path

        self.counter = Counter()
        self.token2idx = {
            "<null>": 0
        }

        self.window_size = window_size
        self.half_of_window_size = int(self.window_size / 2)
        self.min_occurences = minimal_word_occurences

        self.sentences_len = 0

        if dataset is not None and token2idx is not None:
            self.ds = dataset
            self.token2idx = token2idx

            print("Tokens: {}".format(len(self.token2idx)))
        else:
            with open(corpus_path, encoding="utf8") as f:
                for sentence in tqdm(f, desc="Counting tokens"):
                    self.sentences_len += 1
                    sentence = sentence.replace("\n", "")
                    for token in sentence.split("\t"):
                        self.counter[token.lower()] += 1

            ' build vocab '
            self.vocab = set([token for token in self.counter.keys() if self.counter[token] >= self.min_occurences])

            ' build token2idx '
            for token in tqdm(self.vocab, desc="Building token2idx"):
                self.token2idx[token] = len(self.token2idx)

            print("Tokens: {}".format(len(self.token2idx)))

            self.ds = []
            for tokens in tqdm(self.build_sentences(), desc="Building dataset"):
                self.ds += tokens

            padding = [self.token2idx['<null>'] for _ in range(self.half_of_window_size)]
            self.ds = np.array(padding + self.ds + padding)

    def build_sentences(self):
        with open(self.corpus_path, encoding="utf8") as f:
            for sentence in f:
                sentence = sentence.replace("\n", "").lower()

                sent_splt = sentence.split("\t")
                uniq_sentence = set(sent_splt)

                if uniq_sentence.issubset(self.vocab) and len(sent_splt) >= 4:
                    yield self.numericalize(sent_splt)
                    # yield sent_splt

    def numericalize(self, sentence):
        return [self.token2idx[token] for token in sentence]

    def __getitem__(self, index):
        inputs_l = self.ds[index:index + self.half_of_window_size]
        inputs_r = self.ds[index + self.half_of_window_size + 1:index + 2 * self.half_of_window_size + 1]

        return torch.tensor(np.append(inputs_l, inputs_r)).long(), torch.tensor(self.ds[index + self.half_of_window_size]).long()
        # return [], []

    def __len__(self):
        return len(self.ds) - 2 * self.half_of_window_size
        # return self.ds_len


if __name__ == '__main__':
    DS = DS("../data/postprocessed/corpus_sm.txt", 5, minimal_word_occurences=0)
    print("DS len", len(DS))
    for i in range(len(DS)):
        print("DS GET ITEM 0", DS.__getitem__(i))
