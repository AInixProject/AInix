from ainix_common.parsing.parse_primitives import TypeParserResult
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.tests.testutils import send_result
from ainix_kernel.specialtypes.generic_strings import *
from ainix_kernel.specialtypes.generic_strings import \
    _get_all_symbols, _create_root_types, _create_all_word_parts, _name_for_word_part


def test_word_parts_type_parser():
    tc = TypeContext()
    _create_root_types(tc)
    word_part_type = tc.get_type_by_name(WORD_PART_TYPE_NAME)
    _create_all_word_parts(
        tc, [('foo', True), ("bar", True), ("fo", True), ("!", False)])
    parser: TypeParser = word_part_type.default_type_parser
    result: TypeParserResult = send_result(
        parser.parse_string("foobar", word_part_type), None)
    assert result.get_implementation().name == _name_for_word_part("foo")
    result: TypeParserResult = send_result(
        parser.parse_string("barfo", word_part_type), None)
    assert result.get_implementation().name == _name_for_word_part("bar")
    result: TypeParserResult = send_result(
        parser.parse_string("fobw", word_part_type), None)
    assert result.get_implementation().name == _name_for_word_part("fo")
    result: TypeParserResult = send_result(
        parser.parse_string("Fobw", word_part_type), None)
    assert result.get_implementation().name == _name_for_word_part("fo")
    result: TypeParserResult = send_result(
        parser.parse_string("FObw", word_part_type), None)
    assert result.get_implementation().name == _name_for_word_part("fo")
    result: TypeParserResult = send_result(
        parser.parse_string("!fobw", word_part_type), None)
    assert result.get_implementation().name == _name_for_word_part("!")
    result: TypeParserResult = send_result(
        parser.parse_string("*!fobw", word_part_type), None)
    assert result.get_implementation().name == WORD_PART_TERMINAL_NAME


def test_word_parts():
    tc = TypeContext()
    _create_root_types(tc)
    _create_all_word_parts(
        tc, [('foo', True), ("bar", True), ("fo", True), ("!", False)])
    tc.fill_default_parsers()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("foo", WORD_PART_TYPE_NAME)
    word_part_o = ast.next_node_not_copy
    assert word_part_o.implementation.name == _name_for_word_part("foo")
    mod_type_choice = word_part_o.get_choice_node_for_arg(WORD_PART_MODIFIER_ARG_NAME)
    mod_type_object = mod_type_choice.next_node_not_copy
    assert mod_type_object.implementation.name == MODIFIER_LOWER_NAME
    next_type_choice = word_part_o.get_choice_node_for_arg(WORD_PART_NEXT_ARG_NAME)
    next_part_o = next_type_choice.next_node_not_copy
    assert next_part_o.implementation.name == WORD_PART_TERMINAL_NAME
    ### Unparse
    unparser = AstUnparser(tc)
    result = unparser.to_string(ast)
    assert result.total_string == "foo"
    assert result.node_to_string(word_part_o) == "foo"
    assert result.node_to_string(mod_type_choice) == "foo"
    assert result.node_to_string(mod_type_object) == ""


def test_word_parts_upper():
    tc = TypeContext()
    _create_root_types(tc)
    _create_all_word_parts(
        tc, [('foo', True), ("bar", True), ("fo", True), ("!", False)])
    tc.fill_default_parsers()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("FOO", WORD_PART_TYPE_NAME)
    word_part_o = ast.next_node_not_copy
    assert word_part_o.implementation.name == _name_for_word_part("foo")
    mod_type_choice = word_part_o.get_choice_node_for_arg(WORD_PART_MODIFIER_ARG_NAME)
    mod_type_object = mod_type_choice.next_node_not_copy
    assert mod_type_object.implementation.name == MODIFIER_ALL_UPPER
    next_type_choice = word_part_o.get_choice_node_for_arg(WORD_PART_NEXT_ARG_NAME)
    next_part_o = next_type_choice.next_node_not_copy
    assert next_part_o.implementation.name == WORD_PART_TERMINAL_NAME
    ### Unparse
    unparser = AstUnparser(tc)
    result = unparser.to_string(ast)
    assert result.total_string == "FOO"
    assert result.node_to_string(word_part_o) == "FOO"
    assert result.node_to_string(mod_type_choice) == "FOO"
    assert result.node_to_string(mod_type_object) == ""


def test_word_parts_2():
    tc = TypeContext()
    _create_root_types(tc)
    _create_all_word_parts(
        tc, [('foo', True), ("bar", True), ("fo", True), ("!", False)])
    tc.fill_default_parsers()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("fooBarBaz", WORD_PART_TYPE_NAME)
    word_part_o = ast.next_node_not_copy
    assert word_part_o.implementation.name == _name_for_word_part("foo")
    mod_type_choice = word_part_o.get_choice_node_for_arg(WORD_PART_MODIFIER_ARG_NAME)
    mod_type_object = mod_type_choice.next_node_not_copy
    assert mod_type_object.implementation.name == MODIFIER_LOWER_NAME
    next_type_choice = word_part_o.get_choice_node_for_arg(WORD_PART_NEXT_ARG_NAME)
    next_part_o = next_type_choice.next_node_not_copy
    assert next_part_o.implementation.name == _name_for_word_part("bar")
    ### Unparse
    unparser = AstUnparser(tc)
    result = unparser.to_string(ast)
    assert result.total_string == "fooBar"
    assert result.node_to_string(word_part_o) == "fooBar"
    assert result.node_to_string(mod_type_choice) == "foo"
    assert result.node_to_string(mod_type_object) == ""
    assert result.node_to_string(next_type_choice) == "Bar"
    assert result.node_to_string(next_part_o) == "Bar"
    assert result.node_to_string(next_part_o) == "Bar"
