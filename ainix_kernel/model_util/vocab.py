from abc import ABC, abstractmethod
import collections
from typing import Iterable, Type, List, Union
from collections import defaultdict

from ainix_common.parsing.typecontext import AInixType, AInixObject, TypeContext
from ainix_common.parsing.model_specific import parse_constants
import torch
from ainix_common.parsing.ast_components import AstObjectChoiceSet
from ainix_common.parsing.stringparser import StringParser
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_common.parsing.model_specific.tokenizers import Tokenizer
from typing import Hashable, TypeVar, Generic
import numpy as np
import typing

T = TypeVar('T')


class Vocab(Generic[T]):
    """Defines an interface for vocabs. Vocabs are essentially just two way mappings
    between hashables and an integer in the set {0,1,2... num_elements_in_vocab - 1}"""
    @abstractmethod
    def token_to_index(self, token: T) -> int:
        pass

    @abstractmethod
    def index_to_token(self, index: int) -> T:
        pass

    @abstractmethod
    def torch_indices_to_tokens(self, indices: torch.LongTensor) -> np.array:
        """Converts arbitrary shape long tensor of indices into an array of vocab type"""
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def extend(self, v, sort=False):
        pass

    @abstractmethod
    def token_seq_to_indices(
        self,
        sequence: typing.Sequence[Hashable],
        as_torch=True
    ):
        pass

    @abstractmethod
    def items(self):
        raise NotImplemented()

    def get_save_state_dict(self):
        raise NotImplemented()


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
                 specials=(parse_constants.UNK, parse_constants.PAD)):
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        self.itos: List[T] = list(specials)
        try:
            self.unk_index = self.itos.index(parse_constants.UNK)
        except ValueError:
            self.unk_index = None


        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        len_to_use = min(max_size or 9e9, len(words_and_frequencies))
        # TODO (DNGros): if was a cool kid would bisect to find point where freq chnages
        self.itos = np.array([word
                              for word, freq in words_and_frequencies[:len_to_use]
                              if freq >= min_freq])
        self._finish_init()

    def _finish_init(self):
        """Finishes initing an object. Used to calculate stuff that needs to happen
        both for initing and when restoring from a saved_state"""
        # stoi is simply a reverse dict for itos
        self.stoi = defaultdict(lambda x: self.unk_index)
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectorized_stoi = np.vectorize(self.token_to_index)

    def __len__(self):
        return len(self.itos)

    def token_to_index(self, token: T) -> int:
        return self.stoi.get(token, self.unk_index)

    def extend(self, v: 'CounterVocab', sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    def items(self):
        return enumerate(self.itos)

    def index_to_token(self, index: int) -> T:
        return self.itos[index]

    def torch_indices_to_tokens(self, indices: torch.LongTensor) -> np.array:
        return self.itos[indices.numpy()]

    def token_seq_to_indices(
        self,
        sequence: typing.Sequence[T],
        as_torch=True
    ):
        indices = self.vectorized_stoi(sequence)
        if as_torch:
            return torch.from_numpy(indices)
        return indices

    def get_save_state_dict(self):
        return {"itos": self.itos, 'unk_index': self.unk_index, "version": 0}

    @classmethod
    def create_from_save_state_dict(cls, save_state: dict) -> 'CounterVocab':
        instance = cls.__new__(cls)
        instance.itos = save_state['itos']
        instance.unk_index = save_state['unk_index']
        instance._finish_init()
        return instance


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


class TypeContextWrapperVocab(Vocab):
    """A vocab that just wraps all the objects and the types in a TypeContext as
    tokens. It allows for mapping of objects and types into indicies. This
    functionallity is not just built into a type context because vocabs have
    use some torch methods and we are trying to keep ainix_common non-dependent
    on pytorch (I don't know this might be not really necessary though...)."""
    def __init__(self, type_context: TypeContext):
        self.itos: typing.Sequence[Union[AInixType, AInixObject]] = \
            np.array(list(type_context.get_all_objects()) + list(type_context.get_all_types()))
        self._finish_init()

    def _finish_init(self):
        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vectorized_stoi = np.vectorize(self.token_to_index)

    def token_to_index(self, token: Union[AInixObject, AInixType]) -> int:
        return self.stoi[token]

    def index_to_token(self, index: int) -> Union[AInixObject, AInixType]:
        return self.itos[index]

    def torch_indices_to_tokens(self, indices: torch.LongTensor) -> np.array:
        """Converts arbitrary shape long tensor of indices into an array of vocab type"""
        return self.itos[indices.numpy()]

    def __len__(self):
        return len(self.itos)

    def items(self):
        return enumerate(self.itos)

    def extend(self, v, sort=False):
        raise NotImplemented("I should be less lazy...")

    def token_seq_to_indices(
        self,
        sequence: typing.Sequence[Union[AInixObject, AInixType]],
        as_torch=True
    ):
        indices = self.vectorized_stoi(sequence)
        if as_torch:
            return torch.from_numpy(indices)
        return indices

    def get_save_state_dict(self) -> dict:
        return {
            "version": 0,
            "name": "TypeContextWrapperVocab",
            "itos": [(isinstance(item, AInixType), item.name)
                     for item in self.itos]
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        save_state: dict,
        new_type_context: TypeContext
    ) -> 'TypeContextWrapperVocab':
        instance = cls.__new__(cls)
        instance.itos = np.array([
            new_type_context.get_type_by_name(name) if is_type
            else new_type_context.get_object_by_name(name)
            for is_type, name in save_state['itos']
        ])
        instance._finish_init()
        return instance


def make_vocab_from_example_store(
    exampe_store: ExamplesStore,
    x_tokenizer: Tokenizer,
    y_tokenizer: Tokenizer,
    x_vocab_builder: VocabBuilder = None,
    y_vocab_builder: VocabBuilder = None
) -> typing.Tuple[Vocab, Vocab]:
    """Creates an x and y vocab based off all examples in an example store.

    Args:
        exampe_store: The example store to build from
        x_tokenizer: The tokenizer to use for x queries
        y_tokenizer: The tokenizer to use for y queries
        x_vocab_builder: The builder for the kind of x vocab we want to produce.
            If None, it just picks a reasonable default.
        y_vocab_builder: The builder for the kind of y vocab we want to produce.
            If None, it just picks a reasonable default.
    Returns:
        Tuple of the new (x vocab, y vocab).
    """
    if x_vocab_builder is None:
        x_vocab_builder = CounterVocabBuilder(min_freq=1)
    if y_vocab_builder is None:
        y_vocab_builder = CounterVocabBuilder()

    already_done_ys = set()
    parser = StringParser(exampe_store.type_context)
    for example in exampe_store.get_all_examples():
        x_vocab_builder.add_sequence(x_tokenizer.tokenize(example.xquery)[0])
        if example.ytext not in already_done_ys:
            ast = parser.create_parse_tree(example.ytext, example.ytype)
            y_tokens, _ = y_tokenizer.tokenize(ast)
            y_vocab_builder.add_sequence(y_tokens)
    return x_vocab_builder.produce_vocab(), y_vocab_builder.produce_vocab()


def make_vocab_from_example_store_and_type_context(
    example_store: ExamplesStore,
    x_tokenizer: Tokenizer,
    x_vocab_builder: VocabBuilder = None
) -> typing.Tuple[Vocab, TypeContextWrapperVocab]:
    """
    Like the above method, except this one doesn't do special tokenizes to get
    a y vocab and instead just grabs everything from the example_store's type
    context.
    Args:
        example_store: The example store to generate x values from
        x_tokenizer: The tokenizer to generate the tokens we will put in the x vocab
        x_vocab_builder: A builder the x_tokenizer

    Returns:
        The x and y vocab
    """
    if x_vocab_builder is None:
        x_vocab_builder = CounterVocabBuilder(min_freq=1)

    for example in example_store.get_all_examples():
        x_vocab_builder.add_sequence(x_tokenizer.tokenize(example.xquery)[0])
    y_vocab = TypeContextWrapperVocab(example_store.type_context)
    return x_vocab_builder.produce_vocab(), y_vocab


def are_indices_valid(
    indices: torch.Tensor,
    vocab: Vocab[Union[AInixType, AInixObject]],
    valid_set: AstObjectChoiceSet
):
    """Checks some set of indices into a vocab against a AstSet. Returns a tensor
    with value of 1 where known valid and 0 otherwise"""
    objects = vocab.torch_indices_to_tokens(indices)
    valid_func = np.vectorize(
        lambda n: 1 if valid_set.is_known_choice(n.name) else 0,
        otypes='f')
    return torch.from_numpy(valid_func(objects))
