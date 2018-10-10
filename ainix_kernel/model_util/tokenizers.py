from abc import ABC, abstractmethod
from typing import Iterable, Generator, List, Tuple, Hashable, Union

from ainix_common.parsing.typecontext import AInixObject, AInixType
from ainix_kernel import constants
from ainix_common.parsing.ast_components import AstNode, ObjectNode, ObjectChoiceNode
from ainix_common.parsing import ast_components
import numpy as np


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

class NonAsciiTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], None]:
        """Takes in a string and outputs a tokenization splitting at non-ascii boundries"""
        # TODO (DNGros): Maybe memoize this
        if not isinstance(to_tokenize, str):
            raise ValueError(f"NonAsciiTokenizer expects string inputs. Got {to_tokenize}")
        out = [[]]
        for c in to_tokenize:
            if not (c <= 'z' and c >= 'A'):
                if out[-1]:
                    out.append([])

                if c == " ":
                    out[-1].append(constants.SPACE)
                else:
                    out[-1].append(c)
                out.append([])
            else:
                out[-1].append(c)
        out = ["".join(toklist) for toklist in out if len(toklist) >= 1]
        return out, None

class SpaceTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], None]:
        return to_tokenize.split(), None

class AstStringTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: AstNode) -> Tuple[List[str], List[AstNode]]:
        out_tokens = []
        out_nodes = []
        for node in to_tokenize.depth_first_iter():
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
        for node in to_tokenize.depth_first_iter():
            if isinstance(node, ObjectNode):
                out_tokens.append(node.implementation)
            elif isinstance(node, ObjectChoiceNode):
                out_tokens.append(node.type_to_choose)
            else:
                raise ValueError(f"Unrecognized node {node}")
            out_nodes.append(node)
        return out_tokens, out_nodes
