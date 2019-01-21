from builtin_types.otherdata.stackexchange.split_stacke_data import *
import pytest


def test_longest_code_pre():
    assert longest_code_pre_len("sdf <pre><code>abc</code></pre>") == 3


def test_longest_code_pre2():
    assert longest_code_pre_len(""""
        sdf <pre><code>abc</code></pre>
        sdnawe d adfas ae <pre><code>ase we</code></pre>
        """) == 6


def test_longest_code_pre4():
    assert longest_code_pre_len("sdf <pre><code>abcsdfab</code></pre> sa") == 8


def test_longest_code_pre3():
    assert longest_code_pre_len("sdf <pre><code>abc\nsdf\nab\n</code></pre> sa") == 11


def test_sentence_split():
    splits = list(sentence_split("hello there. I am Bob."))
    assert splits == ["hello there.", "I am Bob."]


def test_sentence_split2():
    splits = list(sentence_split("hello fo.bar there. I am Bob."))
    assert splits == ["hello fo.bar there.", "I am Bob."]


def test_sentence_split3():
    splits = list(sentence_split("hello there. I am Bob. sda wa"))
    assert splits == ["hello there.", "I am Bob.", "sda wa"]


def test_sentence_split4():
    splits = list(sentence_split("hello fo.bar there. I am Bob! sda wa"))
    assert splits == ["hello fo.bar there.", "I am Bob!", "sda wa"]


def test_sentence_split5():
    splits = list(sentence_split("hello there\n\nbar bas"))
    assert splits == ["hello there", "bar bas"]


def test_sentence_split6():
    splits = list(sentence_split("hello there"))
    assert splits == ["hello there"]


def test_sentence_split7():
    splits = list(sentence_split("It's 93.1. And it wow"))
    assert splits == ["It's 93.1.", "And it wow"]


def test_sentence_split8():
    splits = list(sentence_split("It's as. And it 😮. cool"))
    assert splits == ["It's as.", "And it 😮.", "cool"]


def test_sentence_split9():
    splits = list(sentence_split("This is awesome!! Regex magic"))
    assert splits == ["This is awesome!!", "Regex magic"]


@pytest.mark.skip("Can't handle abbriviations yet")
def test_sentence_split10():
    splits = list(sentence_split("It is e.g. had. Foo bar"))
    assert splits == ["It's is e.g. had. Foo bar"]


@pytest.mark.skip("Can't handle abbriviations yet")
def test_sentence_split11():
    splits = list(sentence_split("Hello Mr. Smith. How are you?"))
    assert splits == ["Hello Mr. Smith", "How are you?"]
