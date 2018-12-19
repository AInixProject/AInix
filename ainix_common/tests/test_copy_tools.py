from ainix_common.parsing.copy_tools import *
from ainix_common.parsing.model_specific.tokenizers import SpaceTokenizer
from ainix_common.parsing.stringparser import StringParser
from ainix_common.tests.toy_contexts import get_toy_strings_context


def test_substring():
    res = string_in_tok_list(
        "hello there",
        StringTokensMetadata(["hello", " there"])
    )
    assert res == (0, 1)


def test_substring2():
    res = string_in_tok_list(
        "hello there",
        StringTokensMetadata(["why", " ", "hello", " ", "there", " ", "friend"])
    )
    assert res == (2, 4)


def test_substring3():
    res = string_in_tok_list(
        "hello", StringTokensMetadata(["why", " ", "hello", " ", "there", " ", "friend"]))
    assert res == (2, 2)


def test_substring4():
    res = string_in_tok_list(
        "ello there", StringTokensMetadata(["why", " ", "hello", " ", "there", " ", "friend"]))
    assert res is None


def test_substring5():
    res = string_in_tok_list(
        "friend yo", StringTokensMetadata(["why", " ", "hello", " ", "there", " ", "friend"]))
    assert res is None


def test_make_copy_ast():
    tc = get_toy_strings_context()
    parser = StringParser(tc)
    unparser = AstUnparser(tc)
    ast = parser.create_parse_tree("TWO foo bar", "ToySimpleStrs")
    unpar_res = unparser.to_string(ast)
    assert unpar_res.total_string == "TWO foo bar"
    tokenizer = SpaceTokenizer()
    in_str = "Hello there foo cow"
    tokens, metadata = tokenizer.tokenize(in_str)
    result = make_copy_versions_of_tree(ast, unparser, metadata)
    toy_str_obj = result.next_node
    assert toy_str_obj.implementation.name == "two_string"

