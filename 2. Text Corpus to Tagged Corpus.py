import spacy

nlp = spacy.load("en_core_web_sm")
from spacy.symbols import ORTH

import os

from tqdm import tqdm


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


def preprocess_pipeline(input_file_path, output_dir, custom_tokenizer_func, threads=4, batch=100):
    ## todo generate docs
    pipe = nlp.pipe([line.replace("\n", "") for line in open(input_file_path).readlines()], disable=["ner"],
                    n_threads=threads, batch_size=batch)

    sentence_tokens_list = []
    sentence_poss_list = []
    combined_list = []

    tokenizer = custom_tokenizer_func(nlp)
    nlp.tokenizer = tokenizer

    for doc in tqdm(pipe, desc='Processing file'):
        for sent in doc.sents:
            poss = []
            tokens = []
            combs = []
            for token in sent:
                poss.append(token.pos_)
                tokens.append(token.text)
                combs.append(token.text+"|"+token.pos_)

            sentence_poss_list.append(poss)
            sentence_tokens_list.append(tokens)
            combined_list.append(combs)

    with open(os.path.join(output_dir, 'corpus_tokens.txt'), 'w', encoding='utf8') as ft:
        for sentence in sentence_tokens_list:
            ft.write("\t".join(sentence) + " \n")

    with open(os.path.join(output_dir, 'corpus_pos.txt'), 'w', encoding='utf8') as fp:
        for sentence in sentence_poss_list:
            fp.write("\t".join(sentence) + " \n")

    with open(os.path.join(output_dir, 'corpus_combined.txt'), 'w', encoding='utf8') as fp:
        for sentence in combined_list:
            fp.write("\t".join(sentence) + " \n")


if __name__ == '__main__':
    preprocess_pipeline(
        os.path.join('data', 'corpus_big.txt'),
        os.path.join('data', 'tagged_data_big'),
        create_custom_tokenizer
    )
