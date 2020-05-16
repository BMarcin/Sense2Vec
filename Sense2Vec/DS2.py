import os

import torch
import torchtext
from torchtext import data
import spacy
from torchtext.datasets import LanguageModelingDataset

nlp = spacy.load("en_core_web_sm")


def create_custom_tokenizer(nlp):
    """
    Create custom tokenizer to tokenize whole special tokens like <number> or <date> as a single one
    :param nlp: spacy NLP object
    :return:
    """
    from spacy import util
    from spacy.tokenizer import Tokenizer
    from spacy.lang.tokenizer_exceptions import TOKEN_MATCH
    prefixes = nlp.Defaults.prefixes + ('^<i>',)
    suffixes = nlp.Defaults.suffixes + ('</i>$',)
    # remove the tag symbols from prefixes and suffixes
    prefixes = list(prefixes)
    prefixes.remove('<')
    prefixes = tuple(prefixes)
    suffixes = list(suffixes)
    suffixes.remove('>')
    suffixes = tuple(suffixes)
    infixes = nlp.Defaults.infixes
    rules = nlp.Defaults.tokenizer_exceptions
    # rules["<number>"]: [{"X": "<number>"}]
    # rules["<date>"]: [{"X": "<date>"}]
    # rules["<quote>"]: [{"X": "<quote>"}]
    # rules["<empty>"]: [{"X": "<empty>"}]
    # rules["<unknown>"]: [{"X": "<unknown>"}]

    token_match = TOKEN_MATCH
    prefix_search = (util.compile_prefix_regex(prefixes).search)
    suffix_search = (util.compile_suffix_regex(suffixes).search)
    infix_finditer = (util.compile_infix_regex(infixes).finditer)
    return Tokenizer(nlp.vocab, rules=rules,
                     prefix_search=prefix_search,
                     suffix_search=suffix_search,
                     infix_finditer=infix_finditer,
                     token_match=token_match)


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