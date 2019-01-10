from abc import ABC, abstractmethod
import attr
from ainix_common.parsing.typecontext import TypeContext, AInixType
from typing import List, Generator, Tuple
from enum import Enum
import hashlib
import math


class DataSplits(Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


DEFAULT_SPLITS = ((.7, DataSplits.TRAIN), (.3, DataSplits.VALIDATION))
SPLIT_TYPE = Tuple[Tuple[float, DataSplits], ...]


def get_split_from_example(x_string: str, y_type: str, splits: SPLIT_TYPE) -> DataSplits:
    """Generates a pseudorandom selection of a split for example as a function of
    its x_string and y_type. This allows for examples to be placed in the same
    split between runs without storing the splits in a file. It also ensures that
    any duplicate x_string example gets placed in the same split

    Args:
        x_string: A string value for this example's x query
        y_type: The type name of root y type for the example
        splits: A tuple of tuppes in the form
            ((probability of being in split, split name), ...)
    """
    DIGEST_BYTES = 8
    MAX_VAL = 2**(8*DIGEST_BYTES)
    hash = hashlib.blake2b(str.encode(x_string + "::" + y_type), digest_size=DIGEST_BYTES)
    hash_int = int(hash.hexdigest(), 16)
    current_ceiling = 0
    for probability, split_name in splits:
        current_ceiling += probability * MAX_VAL
        if hash_int <= math.ceil(current_ceiling):
            return split_name
    raise RuntimeError(f"Unreachable code reached. Splits {splits}")


@attr.s(auto_attribs=True, frozen=True)
class Example:
    example_id: int
    xquery: str
    ytext: str
    xtype: str
    ytype: str
    weight: float
    y_set_id: str
    split: str = None
    yindexable: str = None


class ExamplesStore(ABC):
    """Represents an ABC for an object that can in some form store examples
    and support the necessary operations required to train on the stored
    examples."""
    DEFAULT_X_TYPE = "WordSequence"

    def __init__(self, type_context: TypeContext):
        self.type_context = type_context

    #@abstractmethod
    #def add_example(self, example: Example) -> None:
    #    pass

    @abstractmethod
    def add_many_to_many_with_weighted(
        self,
        x_values: List[str],
        y_values: List[str],
        x_type: str,
        y_type: str,
        weights: List[float],
        splits: SPLIT_TYPE = DEFAULT_SPLITS
    ) -> None:
        pass

    @abstractmethod
    def get_all_examples(self, filter_splits=None) -> Generator[Example, None, None]:
        pass

    @abstractmethod
    def get_examples_from_y_set(self, y_set_id) -> List[Example]:
        pass

    @abstractmethod
    def get_doc_count(self) -> int:
        pass

    def _default_weight(self, i: int, n: int):
        """Gets a default weight for a value. Each value in the sequence
        is half as preferable as the one before it

        Args:
            i : index in the sequence of values (zero indexed)
            n : total number of values in sequence
        """
        if i+1 > n:
            raise ValueError()
        sequence_sum = 2**n-1
        return (2**(n-i-1))/sequence_sum

    def add_many_to_many_default_weight(
        self,
        x_values: List[str],
        y_values: List[str],
        x_type: str,
        y_type: str,
        splits: SPLIT_TYPE = DEFAULT_SPLITS
    ) -> None:
        """Adds several examples with the y_values default weighted."""
        y_count = len(y_values)
        weights = [self._default_weight(i, y_count)
                   for i, y in enumerate(y_values)]
        self.add_many_to_many_with_weighted(x_values, y_values,
                                            x_type, y_type, weights, splits)
