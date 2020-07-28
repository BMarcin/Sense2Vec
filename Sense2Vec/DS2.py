import sys
from collections import Counter

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class DS(Dataset):
    def __init__(self, file_path, window_size, minimal_word_occurences=3, token2idx=None, dataset=None):
        self.sentences = []
        self.unique_tokens = set()
        self.token2idx = {
            '<end>': 0
        }
        self.tokens_counter = Counter()

        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size

        if token2idx is not None:
            self.token2idx = token2idx
        else:
            ' flattening the file to single list '
            for token in tqdm(open(file_path).read().split(), desc='Counting tokens'):
                if token.lower() not in ['\t', '\n']:
                    self.tokens_counter[token.lower()] += 1

            for sentence in tqdm(open(file_path).readlines(), desc='Removing wrong sentences'):
                sentence = sentence.replace("\n", "").split()
                local_tokens = []
                if len(sentence) > self.window_size:
                    for token in sentence:
                        if token.lower() in self.tokens_counter.keys() and \
                                self.tokens_counter[token.lower()] >= minimal_word_occurences:
                            local_tokens.append(token.lower())

                            if token not in self.unique_tokens:
                                self.unique_tokens.add(token.lower())
                        else:
                            break
                    self.sentences.append(local_tokens)

            ' token2idx '
            for token in tqdm(self.unique_tokens, desc="Building token2idx"):
                self.token2idx[token] = len(self.token2idx)

        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = []

            # build dataset
            for sentence in tqdm(self.sentences, desc="Building dataset"):
                self.dataset += self.process_sentence(sentence)

        del  self.sentences

    def process_sentence(self, tokenized_sentence):
        half_of_window = int(self.window_size / 2)
        parts = []

        ' add special tokens '
        tokenized_sentence = ['<end>' for _ in range(half_of_window)] + tokenized_sentence + ['<end>' for _ in
                                                                                              range(half_of_window)]

        for step in range(len(tokenized_sentence) - 2 * half_of_window):
            ' out token '
            out_token = tokenized_sentence[step + half_of_window]

            ' inputs '
            left_side = tokenized_sentence[step:step + half_of_window]
            right_side = tokenized_sentence[step + half_of_window + 1:step + 1 + 2 * half_of_window]

            parts.append(left_side + [out_token] + right_side)

        return parts

    def numericalize(self, part):
        return [self.token2idx[token] for token in part]

    def split_list_to_CBOW(self, part):
        half_of_window = int(self.window_size / 2)

        inputs = part[0:half_of_window] + part[half_of_window + 1:1 + 2 * half_of_window]
        outputs = part[half_of_window]

        return inputs, outputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        numerizalized = self.split_list_to_CBOW(
            self.numericalize(self.dataset[index])
        )

        return np.array(numerizalized[0]), np.array(numerizalized[1])


if __name__ == '__main__':
    DS = DS("../data/postprocessed/corpus_sm.txt", 5, minimal_word_occurences=0)
    print(DS.dataset)
    print(DS.token2idx)
    print("DS len", len(DS))
    for i in range(len(DS)):
        print("DS GET ITEM {}".format(i), DS.__getitem__(i))
