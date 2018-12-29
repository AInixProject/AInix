import pytest

from ainix_common.parsing import loader
from ainix_common.parsing.copy_tools import *
from ainix_common.parsing.model_specific.tokenizers import SpaceTokenizer, NonLetterTokenizer
from ainix_common.parsing.stringparser import StringParser
from ainix_common.parsing.typecontext import TypeContext
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
    ast_set = AstObjectChoiceSet(tc.get_type_by_name("ToySimpleStrs"))
    ast_set.add(ast, True, 1, 1)
    n: ObjectNode = ast.next_node_not_copy
    arg1set = ast_set.get_next_node_for_choice("two_string").next_node. \
        get_arg_set_data(n.as_childless_node()).get_next_node_for_arg("arg1")
    assert arg1set.is_known_choice("foo")
    assert not arg1set.copy_is_known_choice()
    add_copies_to_ast_set(ast, ast_set, unparser, metadata)
    assert n.implementation.name == "two_string"
    assert arg1set.copy_is_known_choice()
    assert arg1set.is_known_choice("foo")


@pytest.fixture(scope="function")
def numbers_type_context():
    type_context = TypeContext()
    loader.load_path(f"builtin_types/generic_parsers.ainix.yaml", type_context, up_search_limit=3)
    loader.load_path(f"builtin_types/numbers.ainix.yaml", type_context, up_search_limit=3)
    type_context.fill_default_parsers()
    return type_context


def test_numbers_copys(numbers_type_context):
    tc = numbers_type_context
    parser = StringParser(tc)
    unparser = AstUnparser(tc)
    ast = parser.create_parse_tree("0", "Number")
    tokenizer = NonLetterTokenizer()
    in_str = "nil"
    tokens, metadata = tokenizer.tokenize(in_str)
    ast_set = AstObjectChoiceSet(tc.get_type_by_name("Number"))
    ast_set.add(ast, True, 1, 1)
    add_copies_to_ast_set(ast, ast_set, unparser, metadata)
    assert not ast_set.copy_is_known_choice()
