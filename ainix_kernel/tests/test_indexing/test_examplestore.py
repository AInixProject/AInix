from ainix_kernel.indexing.examplestore import *
import string, random, itertools
from collections import Counter


def _random_string():
    # ref https://stackoverflow.com/questions/2257441
    char_set = string.digits + string.ascii_letters
    while True:
        yield ''.join(random.sample(char_set * 8, 8))


def test_random_split():
    splits = ((0.7, DataSplits.TRAIN), (0.3, DataSplits.VALIDATION))
    splitter = DataSplitter(splits)
    sample_count = 50000
    data = (splitter.get_split_from_example(rand_str, "type")
            for rand_str in itertools.islice(_random_string(), sample_count))
    data_counts = Counter(data)
    for expected_percent, split in splits:
        assert abs((data_counts[split] / sample_count) - expected_percent) < 0.01
    # do the same thing but varying the type
    data = (splitter.get_split_from_example("thing", rand_str)
            for rand_str in itertools.islice(_random_string(), sample_count))
    data_counts = Counter(data)
    for expected_percent, split in splits:
        assert abs((data_counts[split] / sample_count) - expected_percent) < 0.01


def test_random_split2():
    splits = ((0.7, DataSplits.TRAIN), (0.3, DataSplits.VALIDATION))
    splitter = DataSplitter(splits)
    # make same data gives same results
    sample_count = 1000
    prechosen_strs = list(itertools.islice(_random_string(), sample_count))
    data_1 = tuple((splitter.get_split_from_example(s, "type") for s in prechosen_strs))
    data_2 = tuple((splitter.get_split_from_example(s, "type") for s in prechosen_strs))
    assert data_1 == data_2


def test_random_split3():
    splits = ((0.7, DataSplits.TRAIN), (0.3, DataSplits.VALIDATION))
    splitter = DataSplitter(splits, seed=213)
    # make check seeds repeatable
    sample_count = 100
    prechosen_strs = list(itertools.islice(_random_string(), sample_count))
    data_1 = tuple((splitter.get_split_from_example(s, "type") for s in prechosen_strs))
    splitter = DataSplitter(splits, seed=213)
    data_2 = tuple((splitter.get_split_from_example(s, "type") for s in prechosen_strs))
    assert data_1 == data_2


def test_random_split4():
    splits = ((0.7, DataSplits.TRAIN), (0.3, DataSplits.VALIDATION))
    splitter = DataSplitter(splits, seed=213)
    # make check different seeds -> different results
    sample_count = 100
    prechosen_strs = list(itertools.islice(_random_string(), sample_count))
    data_1 = tuple((splitter.get_split_from_example(s, "type") for s in prechosen_strs))
    splitter = DataSplitter(splits, seed=554)
    data_2 = tuple((splitter.get_split_from_example(s, "type") for s in prechosen_strs))
    assert data_1 != data_2
