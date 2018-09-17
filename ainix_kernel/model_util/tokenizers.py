from abc import ABC, abstractmethod
from typing import Iterable, Generator, List
from ainix_kernel import constants


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, to_tokenize: str) -> List[str]:
        pass


class NonAsciiTokenizer(Tokenizer):
    def tokenize(self, to_tokenize: str) -> List[str]:
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
        return out
