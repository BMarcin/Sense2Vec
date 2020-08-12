from collections import Counter

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np


class DS(Dataset):
    def __init__(self,
                 corpus_path,
                 window_size,
                 dataset_x=None,
                 dataset_y=None,
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

        if dataset_x is not None and dataset_y is not None and token2idx:
            self.ds_x = dataset_x
            self.ds_y = dataset_y
            self.token2idx = token2idx

            print("Tokens: {}".format(len(self.token2idx)))
        else:
            with open(corpus_path) as f:
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

            self.ds_x = []
            self.ds_y = []
            for inputs, output in tqdm(self.build_ds_padded_sentences(), desc="Building dataset"):
                inputs, output = self.numericalize(inputs, output)
                self.ds_x.append(inputs)
                self.ds_y.append(output)
                # out = [0 for _ in range(len(self.token2idx))]
                # out[output] = 1
                # self.ds_y.append(out)

            self.ds_x = np.array(self.ds_x)
            self.ds_y = np.array(self.ds_y)

        # self.out_base = np.array([0 for _ in range(len(self.token2idx))])
        # with open("ds.txt", "w") as f:
        #     print('saving')
        #     for x, y in self.build_ds():
        #         f.write(",".join(x) + "\t\t" + y + "\n")

    def build_ds_padded_sentences(self):
        with open(self.corpus_path) as f:
            for sentence in f:
                sentence = sentence.replace("\n", "").lower()

                sent_splt = sentence.split("\t")

                uniq_sentence = set(sent_splt)

                if uniq_sentence.issubset(self.vocab) and len(sent_splt) >= self.window_size:
                    splitted = sent_splt
                    sentence_splitted = ['<null>' for _ in range(self.half_of_window_size)] + \
                                        splitted + ['<null>' for _ in range(self.half_of_window_size)]

                    index = len(sentence_splitted)
                    while len(splitted) > 0:
                        token = splitted.pop()

                        inputs_left = sentence_splitted[
                                      index - 2 * self.half_of_window_size - 1:index - 2 * self.half_of_window_size + self.half_of_window_size - 1]
                        inputs_right = sentence_splitted[
                                       index - 2 * self.half_of_window_size + self.half_of_window_size:index - 2 * self.half_of_window_size + self.half_of_window_size + self.half_of_window_size]

                        index = index - 1

                        yield inputs_left + inputs_right, token

    def numericalize(self, inputs, output):
        return [self.token2idx[token] for token in inputs], self.token2idx[output]

    def __getitem__(self, index):
        # inputs, output = self.build_ds()
        # inputs, output = self.numericalize(inputs, output)
        # return torch.tensor(inputs).long(), torch.tensor(output).long()
        # out = torch.zeros((len(self.token2idx)))
        # out[self.ds_y[index]] = 1
        return torch.tensor(self.ds_x[index]).long(), torch.tensor(self.ds_y[index]).float()

    def __len__(self):
        return len(self.ds_x)
        # return self.ds_len


if __name__ == '__main__':
    DS = DS("../data/postprocessed/corpus_sm.txt", 5, minimal_word_occurences=0)
    print("DS len", len(DS))
    for i in range(len(DS)):
        print("DS GET ITEM 0", DS.__getitem__(i))
