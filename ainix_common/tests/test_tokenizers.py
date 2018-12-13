from ainix_common.parsing.model_specific.tokenizers import *
from ainix_common.parsing.model_specific import parse_constants


def test_non_letter():
    data = "my name is 'eve'"
    expected = ['my', parse_constants.SPACE, 'name', parse_constants.SPACE, 'is',
                parse_constants.SPACE, "'", 'eve', "'"]
    tokenizer = NonLetterTokenizer()
    result, not_space = tokenizer.tokenize(data)
    assert result == expected
    assert len(result) == len(not_space)
    assert "".join(not_space) == "my name is 'eve'"


def test_batch_tokenize():
    tokenizer = SpaceTokenizer()
    r = tokenizer.tokenize_batch(["a b", "foo bar"])
    assert len(r) == 2
    assert r[0][0] == ["a", "b"]
    r = tokenizer.tokenize_batch(["a b", "foo bar"], take_only_tokens=True)
    assert len(r) == 2
    assert r[0] == ["a", "b"]
    assert r[1] == ["foo", "bar"]

def test_pad():
    pad = "PAD"
    out = add_str_pads([
        ["foo", "bar"],
        ["fo", "bax", "da"],
        ["sd"]
    ], pad_with=pad)
    assert out == [
        ["foo", "bar", pad],
        ["fo", "bax", "da"],
        ["sd", pad, pad]
    ]

