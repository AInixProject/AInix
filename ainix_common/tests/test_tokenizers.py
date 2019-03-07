from ainix_common.parsing.model_specific.tokenizers import *
from ainix_common.parsing.model_specific import parse_constants
import pytest
import unittest.mock


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
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("a", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["a"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, None]
    assert metad.joinable_tokens_pos_to_actual == [1]


def test_mod_word_piece_tokenizer_two_tok():
    tokenizer = ModifiedWordPieceTokenizer(["ab", "bc", "a", "c"])
    moded_tokens, metad = tokenizer.tokenize("abc")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("ab", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("c", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["ab", "c"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, 1, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2]


def test_mod_word_piece_tokenizer_space():
    tokenizer = ModifiedWordPieceTokenizer(["ab", "bc", "a", "c"])
    moded_tokens, metad = tokenizer.tokenize("abc a")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("ab", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("c", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedStringToken("a", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["ab", "c", " ", "a"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, 1, 3, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2, None, 3]


def test_mod_word_piece_tokenizer_cap():
    tokenizer = ModifiedWordPieceTokenizer(["ab", "bc", "a"])
    moded_tokens, metad = tokenizer.tokenize("A")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("a", CasingModifier.SINGLE_CHAR_UPPER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["A"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, None]
    assert metad.joinable_tokens_pos_to_actual == [1]


def test_mod_word_piece_tokenizer_cap2():
    tokenizer = ModifiedWordPieceTokenizer(["ab", "bc", "a"])
    moded_tokens, metad = tokenizer.tokenize("AB")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("ab", CasingModifier.ALL_UPPER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["AB"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, None]
    assert metad.joinable_tokens_pos_to_actual == [1]


def test_mod_word_piece_tokenizer_cap3():
    tokenizer = ModifiedWordPieceTokenizer(["ab", "bc", "a"])
    moded_tokens, metad = tokenizer.tokenize("Ab")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("ab", CasingModifier.FIRST_UPPER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["Ab"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, None]
    assert metad.joinable_tokens_pos_to_actual == [1]


def test_mod_word_piece_tokenizer_unk():
    tokenizer = ModifiedWordPieceTokenizer(["ab", "bc", "a"])
    moded_tokens, metad = tokenizer.tokenize("dab")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken(parse_constants.UNK, CasingModifier.CASELESS,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("ab", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["d", "ab"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, 1, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2]


@unittest.mock.patch("ainix_common.parsing.model_specific.tokenizers.looks_like_a_file",
                     lambda x: x == "a1234c")
def test_mod_word_piece_tokeizer_merge1():
    tokenizer = ModifiedWordPieceTokenizer(["a", "1", "2", "3", "4", "c"], merge_long_files=True)
    moded_tokens, metad = tokenizer.tokenize("a1234c")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("a", CasingModifier.LOWER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        MOD_TOK_FOR_MERGE,
        ModifiedStringToken("c", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["a", "1234", "c"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, 1, 2, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2, 3]


@unittest.mock.patch("ainix_common.parsing.model_specific.tokenizers.looks_like_a_file",
                     lambda x: x == "a1234c")
def test_mod_word_piece_tokeizer_merge2():
    tokenizer = ModifiedWordPieceTokenizer(
        ["a", "1", "2", "3", "4", "c", "b"], merge_long_files=True)
    moded_tokens, metad = tokenizer.tokenize("a1234c ba")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("a", CasingModifier.LOWER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        MOD_TOK_FOR_MERGE,
        ModifiedStringToken("c", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedStringToken("b", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("a", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["a", "1234", "c", " ", "b", "a"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, 1, 2, 4, 5, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2, 3, None, 4, 5]


@unittest.mock.patch("ainix_common.parsing.model_specific.tokenizers.looks_like_a_file",
                     lambda x: x == "a1234c")
def test_mod_word_piece_tokeizer_merge3():
    tokenizer = ModifiedWordPieceTokenizer(
        ["a", "1", "2", "3", "4", "c", "b"], merge_long_files=True)
    moded_tokens, metad = tokenizer.tokenize("ab a1234c")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("a", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("b", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedStringToken("a", CasingModifier.LOWER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        MOD_TOK_FOR_MERGE,
        ModifiedStringToken("c", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["a", "b", " ", "a", "1234", "c"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, 1, 3, 4, 5, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2, None, 3, 4, 5]


@unittest.mock.patch("ainix_common.parsing.model_specific.tokenizers.looks_like_a_file",
                     lambda x: x == "a1234c" or x == "a12c")
def test_mod_word_piece_tokeizer_merge4():
    tokenizer = ModifiedWordPieceTokenizer(
        ["a", "1", "2", "3", "4", "c", "b"], merge_long_files=True)
    moded_tokens, metad = tokenizer.tokenize("a1234c ba a12c")
    assert moded_tokens == [
        ModifiedWordPieceTokenizer.SOS_TOK,
        ModifiedStringToken("a", CasingModifier.LOWER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        MOD_TOK_FOR_MERGE,
        ModifiedStringToken("c", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedStringToken("b", CasingModifier.LOWER, WhitespaceModifier.AFTER_SPACE_OR_SOS),
        ModifiedStringToken("a", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedStringToken("a", CasingModifier.LOWER,
                            WhitespaceModifier.AFTER_SPACE_OR_SOS),
        MOD_TOK_FOR_MERGE,
        ModifiedStringToken("c", CasingModifier.LOWER, WhitespaceModifier.NOT_AFTER_SPACE),
        ModifiedWordPieceTokenizer.EOS_TOK
    ]
    assert metad.joinable_tokens == ["a", "1234", "c", " ", "b", "a", " ", "a", "12", "c"]
    assert metad.actual_pos_to_joinable_pos == [None, 0, 1, 2, 4, 5, 7, 8, 9, None]
    assert metad.joinable_tokens_pos_to_actual == [1, 2, 3, None, 4, 5, None, 6, 7, 8]
