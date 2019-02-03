from ainix_common.parsing.model_specific.tokenizers import *
from ainix_common.parsing.model_specific import parse_constants
import pytest


def test_non_letter():
    data = "my name is 'eve'"
    expected = ['my', parse_constants.SPACE, 'name', parse_constants.SPACE, 'is',
                parse_constants.SPACE, "'", 'eve', "'"]
    tokenizer = NonLetterTokenizer()
    result, metadata = tokenizer.tokenize(data)
    assert result == expected
    assert len(result) == len(metadata.joinable_tokens)
    assert "".join(metadata.joinable_tokens) == "my name is 'eve'"
    assert metadata.joinable_tokens_pos_to_actual == [0, 1, 2, 3, 4, 5, 6, 7, 8]


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
    out, orig_lengths = add_str_pads([
        ["foo", "bar"],
        ["fo", "bax", "da"],
        ["sd"]
    ], pad_with=pad)
    assert out == [
        ["foo", "bar", pad],
        ["fo", "bax", "da"],
        ["sd", pad, pad]
    ]
    assert orig_lengths == [2, 3, 1]


def test_space_tokenizer():
    tokenizer = SpaceTokenizer()
    str = "hello there you"
    tokens, metadata = tokenizer.tokenize(str)
    assert tokens == ["hello", "there", "you"]
    assert "".join(metadata.joinable_tokens) == str
    assert metadata.joinable_tokens_pos_to_actual == [0, None, 1, None, 2]
    assert metadata.actual_pos_to_joinable_pos == [0, 2, 4]


def test_space_tokenizer2():
    tokenizer = SpaceTokenizer()
    str = "hello"
    tokens, metadata = tokenizer.tokenize(str)
    assert tokens == ["hello"]
    assert "".join(metadata.joinable_tokens) == str
    assert metadata.joinable_tokens_pos_to_actual == [0]



def test_mod_word_piece_tokenizer():
    tokenizer = ModifiedWordPieceTokenizer(["ab", "bc", "a"])
    moded_tokens, metad = tokenizer.tokenize("a")
    assert len(moded_tokens) == 3
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("a", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
