from ainix_kernel.model_util.stop_words import *
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq


def test_stop_word_mask():
    tokenizer = ModifiedWordPieceTokenizer(
        ["foo", "bar", "baz", "the"])
    mask = get_non_stop_word_mask(*tokenizer.tokenize("foo the bar"), stop_words={"the"})
    assert torch_epsilon_eq(
        mask,
        [1, 1, 0, 1, 1]
    )


def test_stop_word_mask2():
    tokenizer = ModifiedWordPieceTokenizer(
        ["foo", "bar", "baz", "the", "a", "b", "c"])
    mask = get_non_stop_word_mask(*tokenizer.tokenize("foo thea bar"), stop_words={"the"})
    assert torch_epsilon_eq(
        mask,
        [1, 1, 1, 1, 1, 1]
    )


def test_stop_word_mask3():
    tokenizer = ModifiedWordPieceTokenizer(
        ["foo", "bar", "baz", "the", "a", "b", "c"])
    mask = get_non_stop_word_mask(*tokenizer.tokenize("foo the bar a"), stop_words={"the", "a"})
    assert torch_epsilon_eq(
        mask,
        [1, 1, 0, 1, 0, 1]
    )


def test_stop_word_mask4():
    tokenizer = ModifiedWordPieceTokenizer(
        ["foo", "bar", "baz", "the", "a", "b", "c"])
    mask = get_non_stop_word_mask(*tokenizer.tokenize("foo a ab"), stop_words={"the", "ab"})
    assert torch_epsilon_eq(
        mask,
        [1, 1, 1, 0, 0, 1]
    )


def test_stop_word_mask5():
    tokenizer = ModifiedWordPieceTokenizer(
        ["foo", "bar", "baz", "the"])
    mask = get_non_stop_word_mask(*tokenizer.tokenize("foo THE bar"), stop_words={"the"})
    assert torch_epsilon_eq(
        mask,
        [1, 1, 0, 1, 1]
    )


def test_stop_word_mask6():
    tokenizer = ModifiedWordPieceTokenizer(
        ["foo", "bar", "baz", "the"])
    mask = get_non_stop_word_mask(*tokenizer.tokenize("foo the bar"),
                                  stop_words={"the"}, pad_to_len=8)
    assert torch_epsilon_eq(
        mask,
        [1, 1, 0, 1, 1, 0, 0, 0]
    )
