"""
Prepare vocabulary and initial word vectors.
"""

import json
import pickle
import argparse
from collections import Counter
from typing import Any, List, Tuple, NamedTuple

# A type alias for JSON.

JSON = Any
Arguments = argparse.Namespace
DefaultSpecials = ['<pad>', '<unk>']


class VocabHelp(object):
    # noinspection PyDefaultArgument
    def __init__(self, counter: Counter, specials=DefaultSpecials):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])

        # words_and_frequencies is a list of tuple
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        return self.stoi == other.stoi and self.itos == other.itos

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str) -> 'VocabHelp':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path: str) -> None:
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description='Prepare vocab.')
    parser.add_argument('--data_dir', default='../data/D1/res16', help='data directory.')
    parser.add_argument('--vocab_dir', default='../data/D1/res16', help='Output vocab directory.')
    parser.add_argument('--lower', default=False, help='If specified, lowercase all words.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'

    # output files
    # token
    # TODO: unused
    # vocab_tok_file = args.vocab_dir + '/vocab_tok.vocab'
    # position
    vocab_post_file = args.vocab_dir + '/vocab_post.vocab'
    # deprel
    vocab_deprel_file = args.vocab_dir + '/vocab_deprel.vocab'
    # postag
    vocab_postag_file = args.vocab_dir + '/vocab_postag.vocab'
    # syn_post
    vocab_synpost_file = args.vocab_dir + '/vocab_synpost.vocab'

    # load files
    print("loading files...")
    train: TokenData = load_tokens(train_file)
    dev: TokenData = load_tokens(dev_file)
    test: TokenData = load_tokens(test_file)

    # lower tokens
    if args.lower:
        train.tokens, dev.tokens, test.tokens = [
            [t.lower() for t in tokens]
            for tokens
            in (train.tokens, dev.tokens, test.tokens)
        ]

    # counters
    token_counter = Counter(train.tokens + dev.tokens + test.tokens)
    deprel_counter = Counter(train.deprel + dev.deprel + test.deprel)
    postag_counter = Counter(train.postag + dev.postag + test.postag)
    # TODO: unused
    # postag_ca_counter = Counter(train_postag_ca + dev_postag_ca + test_postag_ca)
    deprel_counter['self'] = 1

    max_len = max(train.max_len, dev.max_len, test.max_len)
    post_counter = Counter(list(range(0, max_len)))
    syn_post_counter = Counter(list(range(0, 5)))

    # build vocab
    print("building vocab...")
    token_vocab = VocabHelp(token_counter, specials=DefaultSpecials)
    post_vocab = VocabHelp(post_counter, specials=DefaultSpecials)
    deprel_vocab = VocabHelp(deprel_counter, specials=DefaultSpecials)
    postag_vocab = VocabHelp(postag_counter, specials=DefaultSpecials)
    syn_post_vocab = VocabHelp(syn_post_counter, specials=DefaultSpecials)
    print("token_vocab: {}, post_vocab: {}, syn_post_vocab: {}, deprel_vocab: {}, postag_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(syn_post_vocab), len(deprel_vocab), len(postag_vocab)))

    print("dumping to files...")
    # TODO: unused
    # token_vocab.save_vocab(vocab_tok_file)
    post_vocab.save_vocab(vocab_post_file)
    deprel_vocab.save_vocab(vocab_deprel_file)
    postag_vocab.save_vocab(vocab_postag_file)
    syn_post_vocab.save_vocab(vocab_synpost_file)
    print("all done.")


class TokenData(NamedTuple):
    tokens: List[str]
    deprel: List[str]
    postag: List[Tuple]
    postag_ca: List[str]
    max_len: int


def load_tokens(filename: str) -> TokenData:
    with open(filename) as infile:
        data: JSON = json.load(infile)
        tokens: List[str] = []
        deprel: List[str] = []
        postag: List[Tuple] = []
        postag_ca: List[str] = []

        max_len: int = 0
        for d in data:
            sentence: str = d['sentence']
            sentence_words = sentence.split()
            tokens.extend(sentence_words)
            deprel.extend(d['deprel'])
            postag_ca.extend(d['postag'])
            # postag.extend(d['postag'])
            n = len(d['postag'])
            tmp_pos = []
            for i in range(n):
                for j in range(n):
                    tup = tuple(sorted([d['postag'][i], d['postag'][j]]))
                    tmp_pos.append(tup)
            postag.extend(tmp_pos)
            max_len = max(len(sentence_words), max_len)

    print("{} tokens from {} examples loaded from {}.".format(
        len(tokens), len(data), filename))
    return TokenData(tokens, deprel, postag, postag_ca, max_len)


if __name__ == '__main__':
    main()
