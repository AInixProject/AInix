from abc import ABC, abstractmethod
from typecontext import TypeContext, AInixType
from typing import List


class ExamplesStore(ABC):
    DEFAULT_X_TYPE = "WordSequence"
    def __init__(self, type_context: TypeContext):
        pass

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
    ) -> None:
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
        y_type: str
    ) -> None:
        """Adds several examples with the y_values default weighted."""
        y_count = len(y_values)
        weights = [self._default_weight(i, y_count)
                   for i, y in enumerate(y_values)]
        self.add_many_to_many_with_weighted(x_values, y_values,
                                            x_type, y_type, weights)

