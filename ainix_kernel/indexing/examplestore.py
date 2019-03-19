from abc import ABC, abstractmethod
import attr
from ainix_common.parsing.typecontext import TypeContext, AInixType
from typing import List, Generator, Tuple
import enum
import hashlib
import math


@enum.unique
class DataSplits(enum.IntEnum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


DEFAULT_SPLITS = ((.7, DataSplits.TRAIN), (.3, DataSplits.VALIDATION))
SPLIT_PROPORTIONS_TYPE = Tuple[Tuple[float, DataSplits], ...]


def get_split_from_example(
    x_string: str,
    y_type: str,
    splits: SPLIT_PROPORTIONS_TYPE
) -> DataSplits:
    """Generates a pseudorandom selection of a split for example as a function of
    its x_string and y_type. This allows for examples to be placed in the same
    split between runs without storing the splits in a file. It also ensures that
    any duplicate x_string example gets placed in the same split

    Args:
        x_string: A string value for this example's x query
        y_type: The type name of root y type for the example
        splits: A tuple of tuppes in the form
            ((probability of being in split, split class), ...)
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
class XValue:
    """An x example which is associate with a certain set to of valid
    interpretation in a Y type.

    Args:
        example_id: A unique integer identifier of this value
        x_test: A string representation of the utterance.
        y_set_id: A mapping
        x_type: The string identifier for the type this p
        split: the id of what split this value is assigned to
        x_weight: A positive float value which scales how many "examples" this
            example counts as over a "default" example.
    """
    id: int
    x_text: str
    y_set_id: int
    x_type: str
    split: int
    x_weight: float = 1.0


@attr.s(auto_attribs=True, frozen=True)
class YSet:
    id: int
    y_type: str


@attr.s(auto_attribs=True, frozen=True)
class YValue:
    """
    Args:
        y_text: A string representation of the value
        y_type: A string identifier for the type of this value.
        y_set_id: An integer id mapping it to the set this value is a part of.
            This should be sequential. Or in other words, this value will be
            between [0, num_y_values_in_example_store - 1]
        y_preference: A value [0.0, 1.0] which represents the
            preference of this value as a fraction of the most preferable option
            in the YSet. So 0.5 would mean that this value is half as preferable
            as most preferable value in the set.
    """
    id: int
    y_text: str
    y_type: str
    y_set_id: int
    y_preference: float


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
    def add_yset(
        self,
        x_texts: List[str],
        y_texts: List[str],
        x_type: str,
        y_type: str,
        y_preferences: List[float],
        splits: SPLIT_PROPORTIONS_TYPE = DEFAULT_SPLITS
    ) -> None:
        """Essentially used for creating a new YSet.

        Args:
             x_texts: the strs for each x val
             y_texts: the strs for each y val
             x_type: type of x vals
             y_type: type of y_vals
             y_preferences: a preference fraction for every y_text
        """
        pass

    @abstractmethod
    def get_all_examples(self, filter_splits=None) -> Generator[XValue, None, None]:
        pass

    @abstractmethod
    def get_y_values_for_y_set(self, y_set_id) -> List[YValue]:
        pass

    @abstractmethod
    def get_doc_count(self) -> int:
        pass

    @abstractmethod
    def get_example_by_id(self, id: int) -> XValue:
        pass

    @abstractmethod
    def get_x_val_y_type(self, xval: XValue) -> str:
        pass

    def get_non_combined_example_count(self):
        """Gets the count of (x, y) pairs not taking into account y sets.
        For example if three x values each map to a set of two y values, this
        function counts that as 6 examples, rather than just three."""
        raise NotImplementedError()

    def get_y_set_hash(self, y_set_id: int) -> str:
        yvals = self.get_y_values_for_y_set(y_set_id)
        # TODO (DNGros): figure out why actual value hashing not working...
        return hex(abs(hash(tuple(
            [yv.id for yv in yvals]
        ))))


def default_preferences(n: int):
    """Gets a default weight for a value. Each value in the sequence
    is half as preferable as the one before it

    Args:
        n : total number of values in sequence
    """
    return [0.5**i for i in range(n)]


class BasicExampleStore(ExamplesStore):
    """A basic implementation of the ExampleStore interface using builtin python primitives
    an no extra frills."""

    def __init__(self, type_context: TypeContext):
        super().__init__(type_context)
        self._x_vals: List[XValue] = []
        self._y_sets: List[YValue] = []
        self._y_values: List[YValue] = []
        self._y_set_id_to_y_values: List[List[YValue]] = []
        # Used to enforce that we don't have duplicate x values
        self._x_uniqueness_set = set()

    def add_yset(
        self,
        x_texts: List[str],
        y_texts: List[str],
        x_type: str,
        y_type: str,
        y_preferences: List[float],
        splits: SPLIT_PROPORTIONS_TYPE = DEFAULT_SPLITS
    ) -> None:
        if len(y_preferences) != len(y_texts):
            raise ValueError()
        new_y_set = YSet(len(self._y_sets), y_type)
        self._y_sets.append(new_y_set)
        self._y_set_id_to_y_values.append([])
        for y_text, weight in zip(y_texts, y_preferences):
            self.add_y_value(y_text, y_type, new_y_set.id, weight)
        for x_text in x_texts:
            split = get_split_from_example(x_text, y_type, splits)
            self.add_x_value(x_text, new_y_set.id, x_type, split)

    def add_y_value(self, y_text, y_type, y_set_id, preference) -> YValue:
        if y_set_id < 0 or y_set_id >= len(self._y_sets):
            raise ValueError()
        if self._y_sets[y_set_id].y_type != y_type:
            raise ValueError("Attempt to insert a y value into a set of different type")
        if preference < 0 or preference > 1:
            raise ValueError()
        # TODO (DNGros): enforce uniqueness amoung y values in a YSet
        new_val = YValue(len(self._y_values), y_text, y_type, y_set_id, preference)
        self._y_values.append(new_val)
        self._y_set_id_to_y_values[y_set_id].append(new_val)
        return new_val

    def add_x_value(self, x_text, y_set_id, x_type, split, x_weight=1) -> XValue:
        if y_set_id < 0 or y_set_id >= len(self._y_sets):
            raise ValueError()
        uniqueness_tuple = (x_text, x_type, self._y_sets[y_set_id].y_type)
        if uniqueness_tuple in self._x_uniqueness_set:
            raise ValueError(f"Attempt to insert duplicate x value '{x_text}'")
        new_val = XValue(len(self._x_vals), x_text, y_set_id, x_type, split, x_weight)
        self._x_vals.append(new_val)
        self._x_uniqueness_set.add(uniqueness_tuple)
        return new_val

    def get_all_examples(
        self,
        only_from_splits: Tuple[DataSplits, ...] = None
    ) -> Generator[XValue, None, None]:
        yield from (xv for xv in self._x_vals
                    if only_from_splits is None or xv.split in only_from_splits)

    def get_y_values_for_y_set(self, y_set_id: int) -> List[YValue]:
        return sorted(list(self._y_set_id_to_y_values[y_set_id]), key=lambda v: v.y_preference)

    def get_x_val_y_type(self, xval: XValue) -> str:
        return self._y_sets[xval.y_set_id].y_type

    def get_doc_count(self) -> int:
        return len(self._x_vals)

    def get_example_by_id(self, id: int) -> XValue:
        return self._x_vals[id]

    def get_non_combined_example_count(self):
        raise NotImplemented()
