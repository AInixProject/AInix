from indexing.examplestore import *
import string, random, itertools
from collections import Counter


def test_random_split():
    splits = ((0.7, DataSplits.TRAIN), (0.3, DataSplits.VALIDATION))
    def random_string():
        # ref https://stackoverflow.com/questions/2257441
        char_set = string.digits + string.ascii_letters
        while True:
            yield ''.join(random.sample(char_set * 8, 8))

    sample_count = 50000
    data = (get_split_from_example(rand_str, "type", splits)
            for rand_str in itertools.islice(random_string(), sample_count))
    data_counts = Counter(data)
    for expected_percent, split in splits:
        assert abs((data_counts[split] / sample_count) - expected_percent) < 0.01
    # do the same thing but varying the type
    data = (get_split_from_example("thing", rand_str, splits)
            for rand_str in itertools.islice(random_string(), sample_count))
    data_counts = Counter(data)
    for expected_percent, split in splits:
        assert abs((data_counts[split] / sample_count) - expected_percent) < 0.01
    # make same data gives same results
    sample_count = 1000
    prechosen_strs = list(itertools.islice(random_string(), sample_count))
    data_1 = tuple((get_split_from_example(s, "type", splits) for s in prechosen_strs))
    data_2 = tuple((get_split_from_example(s, "type", splits) for s in prechosen_strs))
    assert data_1 == data_2


