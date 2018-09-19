from abc import ABC, abstractmethod
import typing, collections
from typing import Iterable, Type, List
from collections import defaultdict
from ainix_kernel import constants
import torch


class Vocab:
    @abstractmethod
    def token_to_index(self, token: str) -> int:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def extend(self, v, sort=False):
        pass

    def token_seq_to_indices(self, sequence: Iterable[str]) -> torch.LongTensor:
        return torch.LongTensor([list(map(self.token_to_index, sequence))])


class CounterVocab(Vocab):
    """A vocab that takes is constructed via counter with counts of tokens

    This is based off torchtext vocabs
    Copyright (c) James Bradbury and Soumith Chintala 2016
    https://github.com/pytorch/text/blob/master/torchtext/vocab.py

    Args:
        counter: Counter object holding the frequency of each token
        max_size: The max size of the vocab. If None have no max.
        min_freq: The minium frequency needed to be included in the vocabulary
        specials: The list of special tokens (e.g., padding or eos) that
            will be prepended to the vocabulary in addition to an <unk>
            token.
    """
    def __init__(self, counter: typing.Counter, max_size: int=None, min_freq: int = 1,
                 specials=('<pad>',)):
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        self.itos = list(specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = defaultdict(lambda x: constants.UNK)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def __len__(self):
        return len(self.itos)

    def token_to_index(self, token: str):
        return self.stoi[token]

    def extend(self, v: 'CounterVocab', sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class VocabBuilder(ABC):
    @abstractmethod
    def add_sequence(self, sequence: Iterable[str]):
        pass

    @abstractmethod
    def produce_vocab(self) -> Vocab:
        pass

    @abstractmethod
    def extend_vocab(self):
        pass


class CounterVocabBuilder(VocabBuilder):
    def __init__(self, vocab_to_make: Type[CounterVocab]=CounterVocab, **kwargs):
        self._counter = collections.Counter()
        self.vocab_to_make = vocab_to_make
        self.in_args = kwargs

    def add_sequence(self, sequence: Iterable[str]):
        self._counter.update(sequence)

    def produce_vocab(self) -> Vocab:
        return self.vocab_to_make(self._counter, **self.in_args)

    def extend_vocab(self):
        raise NotImplemented()
