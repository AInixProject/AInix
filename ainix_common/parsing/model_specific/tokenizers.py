"""Classes for tokenizeing a input string. Note this is not used by the actual
string parsers. Rather it can be useful for something like models which wish
to tokenize the input string. It is used in string parsers in order to enable
producing AST's with copying."""
from abc import ABC, abstractmethod
from typing import Iterable, Generator, List, Tuple, Hashable, Union, Optional

import attr
from ainix_common.parsing.typecontext import AInixObject, AInixType
from ainix_common.parsing.model_specific import parse_constants
from ainix_common.parsing.ast_components import AstNode, ObjectNode, ObjectChoiceNode
from ainix_common.parsing import ast_components
import numpy as np
from itertools import chain


class Tokenizer(ABC):
    def __init__(self):
        self._vectorized_tokenize = np.vectorize(self.tokenize)
        self._vectorized_tokenize_first = np.vectorize(lambda x: self.tokenize(x)[0])

    @abstractmethod
    def tokenize(self, to_tokenize) -> Tuple[List[Hashable], List]:
        """

        Args:
            to_tokenize: The object to tokenize. Could be a str, or depending
                on the tokenizer, some other type.

        Returns:
            A tuple. The first element is a list of of the individual tokens.
            The second is metadata about those tokens which could vary depending
            on the parser. It is intended for something like pointers of back to
            the location of the tokens or something like that.
        """
        pass

    def tokenize_batch(self, batch, take_only_tokens: bool = False) -> List:
        """Tokenize multiple values at once

        Args:
            batch: Sequence of objects to tokenize
            take_only_tokens: self.tokenize is expected to return a tuple of
                (tokens, metadata). If this is set to true, then we only take
                the tokens, ignoring the metadata.

        Returns:
            List with self.tokenize applied on every element. The result
            will be tuples if the take_only_tokens is false, and sequence
            if it is true.
        """
        if take_only_tokens:
            return [self.tokenize(s)[0] for s in batch]
        else:
            return [self.tokenize(s) for s in batch]


class StringTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], 'StringTokensMetadata']:
        """Returns Tuple. The first is a list of the actual tokens that could
        go into a vocab. The second is metadata about that. See StringTokensMetadata"""
        raise NotImplemented()


@attr.s(frozen=True)
class StringTokensMetadata:
    """Metadata about the tokens returned by a StringTokenizer

    Args:
        joinable_tokens: A list of tokens which when joined with empty string
            gets back to the origional input string. The tokenization process
            might add in special tokens like "<SPACE>", "<SOS>", or some
            customness for inner-word splitting. This is used for
            something like copying which is only allowed to perform copies
            on token boundries.
        joinable_tokens_pos_to_actual: A mapping from the joinable tokens
            back to original tokens. For example if origional tokens were
            ["<SOS>", "foo", "<SPACE>", "_b", "ar"] joinable tokens might be
            ["foo", " ", "b", "ar"] and the map might be [1, 2, 3, 4]. If
            mapping is None it is considered there is no direct mapping for
            this token and it cannot serve as the start or end of a copy.
        actual_pos_to_joinable_pos: The inverse of the above arg
    """
    joinable_tokens = attr.ib(type=List[str])
    # TODO (DNGros): Change these maps to callable functions. Having them as
    # lists is kinda excessive when sometimes have a closedform map
    joinable_tokens_pos_to_actual = attr.ib(type=List[Optional[int]])
    actual_pos_to_joinable_pos = attr.ib(type=List[Optional[int]])

    @joinable_tokens_pos_to_actual.default
    def fac(self):
        return list(range(len(self.joinable_tokens)))

    @actual_pos_to_joinable_pos.default
    def ofac(self):
        return list(range(len(self.joinable_tokens)))

    def __attrs_post_init__(self):
        assert len(self.joinable_tokens) == len(self.joinable_tokens_pos_to_actual)


class NonLetterTokenizer(StringTokenizer):
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], StringTokensMetadata]:
        """Takes in a string and outputs a tokenization splitting at non-letter boundries"""
        # TODO (DNGros): Maybe memoize this
        if not isinstance(to_tokenize, str):
            raise ValueError(f"NonAsciiTokenizer expects string inputs. Got {to_tokenize}")
        out_tokens = [[]]
        for c in to_tokenize:
            char_is_letter = ('A' <= c <= 'z')
            if not char_is_letter:
                if out_tokens[-1]:
                    out_tokens.append([])

                if c == " ":
                    out_tokens[-1].append(parse_constants.SPACE)
                else:
                    out_tokens[-1].append(c)
                out_tokens.append([])
            else:
                out_tokens[-1].append(c)
        out_tokens = ["".join(toklist) for toklist in out_tokens if len(toklist) >= 1]
        replace_spaces = [x if x != parse_constants.SPACE else " " for x in out_tokens]
        metadata = StringTokensMetadata(replace_spaces)
        return out_tokens, metadata


class SpaceTokenizer(StringTokenizer):
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], StringTokensMetadata]:
        tokens = to_tokenize.split()
        joinable = [tokens[0]] + list(chain.from_iterable(((" ", t) for t in tokens[1:])))
        joinable_to_actual = [0] + list(chain.from_iterable(
            ((None, i + 1) for i in range(len(tokens[1:])))
        ))
        actual_to_joinable = list(range(0, len(tokens)*2, 2))
        return to_tokenize.split(), StringTokensMetadata(
            joinable, joinable_to_actual, actual_to_joinable)


class AstStringTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: AstNode) -> Tuple[List[str], List[AstNode]]:
        out_tokens = []
        out_nodes = []
        for pointer in to_tokenize.depth_first_iter():
            node = pointer.cur_node
            if isinstance(node, ObjectNode):
                out_tokens.append(ast_components.indexable_repr_object(node.implementation.name))
            elif isinstance(node, ObjectChoiceNode):
                out_tokens.append(ast_components.indexable_repr_classify_type(
                    node.get_type_to_choose_name()))
            else:
                raise ValueError(f"Unrecognized node {node}")
            out_nodes.append(node)
        return out_tokens, out_nodes


class AstValTokenizer(Tokenizer):
    def tokenize(
        self,
        to_tokenize: AstNode
    ) -> Tuple[List[Union[AInixObject, AInixType]], List[AstNode]]:
        out_tokens = []
        out_nodes = []
        for pointer in to_tokenize.depth_first_iter():
            node = pointer.cur_node
            if isinstance(node, ObjectNode):
                out_tokens.append(node.implementation)
            elif isinstance(node, ObjectChoiceNode):
                out_tokens.append(node.type_to_choose)
            else:
                raise ValueError(f"Unrecognized node {node}")
            out_nodes.append(node)
        return out_tokens, out_nodes


def add_str_pads(token_seqs: List[List[str]], pad_with=parse_constants.PAD):
    """Add padding tokens to an collection of tokenized values so all the same len

    Returns:
        padded_seqs: new vals but padded.
        origional_lengths
    """
    lengths = list(map(len, token_seqs))
    longest_len = max(lengths)
    pad_val_arr = [pad_with]
    padded_seqs = [existing + pad_val_arr*(longest_len - length)
                   for existing, length in zip(token_seqs, lengths)]
    return padded_seqs, lengths

