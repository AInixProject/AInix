from ainix_common.parsing.copy_tools import *


def test_substring():
    res = string_in_tok_list("hello there", ["hello", " there"])
    assert res == (0, 1)


def test_substring2():
    res = string_in_tok_list("hello there", ["why", " ", "hello", " ", "there", " ", "friend"])
    assert res == (2, 4)


def test_substring3():
    res = string_in_tok_list("hello", ["why", " ", "hello", " ", "there", " ", "friend"])
    assert res == (2, 2)


def test_substring4():
    res = string_in_tok_list("ello there", ["why", " ", "hello", " ", "there", " ", "friend"])
    assert res is None


def test_substring5():
    res = string_in_tok_list("fried yo", ["why", " ", "hello", " ", "there", " ", "friend"])
    assert res is None
