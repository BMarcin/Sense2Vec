import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Sense2Vec.tokenizer import create_custom_tokenizer
import spacy

nlp = spacy.load("en_core_web_sm")


class DS(Dataset):
    def __init__(self, file_path, window_size):
        nlp.tokenizer = create_custom_tokenizer(nlp)

        self.tokens = []
        self.token2idx = {
            '<end>': 0
        }

        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size

        ' flattening the file to single list '
        for doc in tqdm(nlp.pipe(
                open(file_path),
                disable=["ner"],
                batch_size=100,
                n_process=1
        ), desc='Preprocessing'):
            for token in doc:
                if token.text != "\n":
                    self.tokens.append(token.text.lower())

        ' token2idx '
        for token in tqdm(set(self.tokens), desc="Building token2idx"):
            self.token2idx[token] = len(self.token2idx)

        ' fix begging of tokens list '
        self.tokens = ['<end>' for _ in range(int(self.window_size/2))] + self.tokens

        ' build dataset '
        progress = tqdm(total=len(self.tokens), desc='Building dataset')
        self.dataset = self.process_sequence(self.tokens, lambda x: progress.update(x))
        progress.close()

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
        return (ins, outs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        numerizalized = self.split_list_to_CBOW(
            self.numericalize(self.dataset[index])
        )

        return np.array(numerizalized[0]), np.array(numerizalized[1])


if __name__ == '__main__':
    DS = DS("../data/corpus_sm.txt", 7)
    print("DS len", len(DS))
    for i in range(len(DS)):
        print("DS GET ITEM 0", DS.__getitem__(i))
