from ainix_kernel.util.sampling import WeightedRandomChooser
from collections import Counter


def test_rand_choser():
    elements = ["foo", "bar", "baz", "bloop"]
    weights = [1500,5000,3000,500]
    chooser = WeightedRandomChooser(elements, weights)
    samples = [chooser.sample() for i in range(sum(weights))]
    count = Counter(samples)
    
    for e, w in zip(elements, weights):
        assert abs(count[e] - w) < 200
