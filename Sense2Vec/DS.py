import sys
from collections import Counter

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from Sense2Vec.tokenizer import create_custom_tokenizer


class DS(Dataset):
    def __init__(self, file_path, window_size, minimal_word_occurences=3, token2idx=None, dataset=None):
        self.tokens = []
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

            # for doc in tqdm(nlp.pipe(
            #         open(file_path),
            #         disable=["ner"],
            #         batch_size=30000,
            #         n_process=16
            # ), desc='Removing wrong sentences'):
            for sentence in tqdm(open(file_path).readlines(), desc='Removing wrong sentences'):
                sentence = sentence.replace("\n", "").split()
                local_tokens = []
                if len(sentence) > self.window_size:
                    for token in sentence:
                        if token.lower() in self.tokens_counter.keys() and \
                                self.tokens_counter[token.lower()] >= minimal_word_occurences:
                            local_tokens.append(token.lower())
                        else:
                            break
                    self.tokens += local_tokens

            ' token2idx '
            for token in tqdm(set(self.tokens), desc="Building token2idx"):
                self.token2idx[token] = len(self.token2idx)

        if dataset is not None:
            self.dataset = dataset
        else:
            ' fix begging of tokens list '
            self.tokens = ['<end>' for _ in range(int(self.window_size / 2))] + self.tokens

            print("Sample", self.tokens[0:100])

            ' build dataset '
            progress = tqdm(total=len(self.tokens), desc='Building dataset')
            self.dataset = self.process_sequence(self.tokens, lambda x: progress.update(x))
            progress.close()

            del self.tokens

    def process_sequence(self, tokenized_sentence, lambda_fn=None):
        parts = []
        for step in range(0, len(self.tokens) - 1):
            part = tokenized_sentence[step:step + self.window_size]
            part += ['<end>' for _ in range(self.window_size - len(part))]
            parts.append(part)

            if lambda_fn:
                lambda_fn(step)
        return parts

    def numericalize(self, part):
        return [self.token2idx[token] for token in part]

    def split_list_to_CBOW(self, part):
        half = int(self.window_size / 2)
        ins = part[0:half] + part[half + 1:]
        outs = part[half]
        return ins, outs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        numerizalized = self.split_list_to_CBOW(
            self.numericalize(self.dataset[index])
        )

        return np.array(numerizalized[0]), np.array(numerizalized[1])


if __name__ == '__main__':
    DS = DS("../data/postprocessed/corpus_sm.txt", 5, minimal_word_occurences=0)
    print("DS len", len(DS))
    for i in range(len(DS)):
        print("DS GET ITEM 0", DS.__getitem__(i))
