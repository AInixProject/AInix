from ainix_common.parsing.copy_tools import *
from ainix_common.parsing.stringparser import StringParser
from ainix_common.tests.toy_contexts import get_toy_strings_context


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
    res = string_in_tok_list("friend yo", ["why", " ", "hello", " ", "there", " ", "friend"])
    assert res is None


def test_make_copy_ast():
    tc = get_toy_strings_context()
    parser = StringParser(tc)
    unparser = AstUnparser(tc)
    ast = parser.create_parse_tree("TWO foo bar", "ToySimpleStrs")
    unpar_res = unparser.to_string(ast)
    assert unpar_res.total_string == "TWO foo bar"
    make_copy_versions_of_tree(ast, unparser, "Hello there foo sir")
