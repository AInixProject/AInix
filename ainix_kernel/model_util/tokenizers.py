from abc import ABC, abstractmethod
from typing import Iterable, Generator, List, Tuple, Hashable, Union

from ainix_common.parsing.typecontext import AInixObject, AInixType
from ainix_kernel import constants
from ainix_common.parsing.parseast import AstNode, ObjectNode, ObjectChoiceNode
from ainix_common.parsing import parseast


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, to_tokenize) -> Tuple[List[Hashable], List]:
        """Maps a value to a list of tokens and list of metadata about those tokens"""
        pass


class NonAsciiTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: str) -> Tuple[List[str], None]:
        """Takes in a string and outputs a tokenization splitting at non-ascii boundries"""
        # TODO (DNGros): Maybe memoize this
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
                out_tokens.append(parseast.indexable_repr_object(node.implementation.name))
            elif isinstance(node, ObjectChoiceNode):
                out_tokens.append(parseast.indexable_repr_classify_type(
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
