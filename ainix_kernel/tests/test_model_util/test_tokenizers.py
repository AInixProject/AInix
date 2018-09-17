from ainix_kernel.model_util.tokenizers import *
from ainix_kernel import constants


def test_non_ascii():
    data = "my name is 'eve'"
    expected = ['my', constants.SPACE, 'name', constants.SPACE, 'is',
                constants.SPACE, "'", 'eve', "'"]
    tokenizer = NonAsciiTokenizer()
    assert tokenizer.tokenize(data) == expected
