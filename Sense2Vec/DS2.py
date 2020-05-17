import os

import torch
from torchtext import data
from torchtext.datasets import LanguageModelingDataset
from Sense2Vec.tokenizer import create_custom_tokenizer
import spacy

nlp = spacy.load("en_core_web_sm")


class DS2:
    def __init__(self, input_file_path, batch_size, bptt_len, device):
        self.batch_size, self.bptt_len, self.device = batch_size, bptt_len, device

        self.tokenizer = create_custom_tokenizer(nlp)
        self.TEXT = data.Field(lower=True, tokenize=self.tokenize, batch_first=True)
        self.dataset = LanguageModelingDataset(input_file_path, self.TEXT)
        self.TEXT.build_vocab(self.dataset)

    def tokenize(self, text):
        return [tok.text for tok in self.tokenizer(text)]

    def build_iterator(self):
        train_iterator = data.BPTTIterator(
            self.dataset,
            batch_size=self.batch_size,
            bptt_len=self.bptt_len,
            device=self.device,
            repeat=False,
            shuffle=True
        )
        return train_iterator


if __name__ == '__main__':
    ds = DS2(os.path.join("..", "data", "tagged_data", "corpus_tokens.txt"), 2, 3, torch.device("cpu"))
    batches = ds.build_iterator()

    for batch in batches:
        print(batch.text)
        print(batch.target)
        break
