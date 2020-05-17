import spacy
from spacy import util
from spacy.tokenizer import Tokenizer
from spacy.lang.tokenizer_exceptions import TOKEN_MATCH

nlp = spacy.load("en_core_web_sm")


def create_custom_tokenizer(nlp):
    """
    Create custom tokenizer to tokenize whole special tokens like <number> or <date> as a single one
    :param nlp: spacy NLP object
    :return:
    """
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
