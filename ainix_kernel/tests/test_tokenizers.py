from ainix_kernel import constants
from ainix_kernel.tokenizers import *

def test_non_ascii():
    data = "my name is 'eve'"
    expected = ['my', constants.SPACE, 'name', constants.SPACE, 'is',
        constants.SPACE, "'", 'eve', "'"]
    assert nonascii_tokenizer(data) == expected

##############
def test_split_tokenization1():
    inseq = [constants.SOS, "hello", constants.SPACE, "there", constants.EOS]
    expected = [(x,) for x in inseq]
    assert split_tokenization(inseq) == expected, "Failed when no actual grouping"

def test_split_tokenization2():
    inseq = [constants.SOS, "hello", constants.SPACE, "mr", "mrs", constants.EOS]
    expected = [(constants.SOS,), ("hello",), 
            (constants.SPACE,), ("mr", "mrs"), (constants.EOS,)]
    assert split_tokenization(inseq) == expected

def test_split_tokenization3():
    inseq = [constants.SOS, "hello", constants.SPACE, 
            "mr", "mrs", constants.SPACE, "yo", constants.EOS]
    expected = [(constants.SOS,), ("hello",), (constants.SPACE,), 
            ("mr", "mrs"), (constants.SPACE,), ("yo",), (constants.EOS,)]
    assert split_tokenization(inseq) == expected

def test_split_tokenization_no_sos():
    inseq = ["my", constants.SPACE, "name", constants.SPACE, "is", constants.SPACE,
        '"', "bob", '"']
    expected = [("my",), (constants.SPACE,), 
            ("name",), (constants.SPACE,), ("is",), (constants.SPACE,), ('"',"bob",'"')]
    assert split_tokenization(inseq) == expected
