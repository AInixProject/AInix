import random
import bisect
from typing import List, Sequence, TypeVar, Generic

T = TypeVar('T')


class WeightedRandomChooser(Generic[T]):
    """Used for taking a weighted random sample of values

    Args:
        elements: The elements to sample from
        weights: A list of matching len of elements. The values do not have
            sum to 1. A if you have vals [1,3] then then first element will
            be picked 1/4 times.
    """
    def __init__(self, elements: Sequence[T], weights: List[float]):
        if len(elements) != len(weights):
            raise ValueError("Unequal sizes between elements and weights")
        sorted_elements = sorted(zip(weights, elements), key=lambda x: x[0])
        self.weights, self.elements = zip(*sorted_elements)
        self._cum_weights = []
        cum = 0
        for w in self.weights:
            cum += w
            self._cum_weights.append(cum)
        self.weight_sum = cum

    def sample(self) -> T:
        rnd = random.random() * self.weight_sum
        idx = bisect.bisect_right(self._cum_weights, rnd)
        return self.elements[idx]
