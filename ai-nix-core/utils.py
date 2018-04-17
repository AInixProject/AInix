import random
import bisect
import pudb
class WeightedRandomChooser():
    def __init__(self, elements, weights):
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

    def sample(self): 
        rnd = random.random() * self.weight_sum
        idx = bisect.bisect_right(self._cum_weights, rnd)
        return self.elements[idx]
