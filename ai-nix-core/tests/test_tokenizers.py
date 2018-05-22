import constants
from tokenizers import *
def test_split_tokenization1():
    inseq = [constants.SPACE, "hello", constants.SPACE, "there", constants.EOS]
    expected = [(x,) for x in inseq]
    assert split_tokenization(inseq) == expected, "Failed when no actual grouping"

def test_split_tokenization2():
    inseq = [constants.SPACE, "hello", constants.SPACE, "mr", "mrs", constants.EOS]
    expected = [(constants.SPACE,), ("hello",), 
            (constants.SPACE,), ("mr", "mrs"), (constants.EOS,)]
    assert split_tokenization(inseq) == expected

def test_split_tokenization3():
    inseq = [constants.SPACE, "hello", constants.SPACE, 
            "mr", "mrs", constants.SPACE, "yo", constants.EOS]
    expected = [(constants.SPACE,), ("hello",), (constants.SPACE,), 
            ("mr", "mrs"), (constants.SPACE,), ("yo",), (constants.EOS,)]
    assert split_tokenization(inseq) == expected
