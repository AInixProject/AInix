from parsing.typecontext import TypeContext, AInixType
import attr
from typing import List, Dict, Set
from collections import defaultdict
DEFAULT_X_TYPE = "WordSequence"


@attr.s(auto_attribs=True, frozen=True)
class TypePair:
    x_type: str
    y_type: str


@attr.s(auto_attribs=True, frozen=True)
class YString:
    value: str
    weight: float = 1


class ExampleContext:
    def __init__(self, type_context: TypeContext):
        self.type_context = type_context
        # examples go from a type pair -> dict going from x_string -> dict of string to weight
        self.examples: Dict[TypePair, Dict[str, Dict[str, float]]] = \
            defaultdict(dict)

    def add_example(self, x_string: str, y_string: YString, type_pair: TypePair) -> None:
        valid_ys = self.examples[type_pair][x_string]
        if y_string.value in valid_ys:
            valid_ys[y_string.value] += y_string.weight
        else:
            valid_ys[y_string.value] = y_string.weight

    def add_many_to_many_with_weighted(
        self,
        x_values: List[str],
        y_values: List[YString],
        type_pair: TypePair
    ) -> None:
        for x in x_values:
            for y in y_values:
                self.add_example(x, y, type_pair)

    def _default_weight(self, i: int, n: int):
        """Get's a default weight for a value. Each value in the sequence
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
        type_pair: TypePair
    ) -> None:
        """Adds several examples with the y_values default weighted."""
        y_count = len(y_values)
        y_strings = [YString(y, self._default_weight(i, y_count))
                     for i, y in enumerate(y_values)]
        self.add_many_to_many_with_weighted(x_values, y_strings, type_pair)
