from ainix_common.parsing import loader
from ainix_common.parsing.parse_primitives import TypeParserResult
from ainix_common.parsing.stringparser import StringParser, AstUnparser, UnparseResult
from ainix_common.tests.testutils import send_result
from ainix_kernel.specialtypes import generic_strings
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
    tc.finalize_data()
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
    pointers = list(ast.depth_first_iter())
    assert ast == pointers[0].cur_node
    assert result.pointer_to_string(pointers[0]) == "foo"
    assert word_part_o == pointers[1].cur_node
    assert result.pointer_to_string(pointers[1]) == "foo"
    assert mod_type_choice == pointers[2].cur_node
    assert result.pointer_to_string(pointers[2]) == "foo"
    assert mod_type_object == pointers[3].cur_node
    assert result.pointer_to_string(pointers[3]) == ""


def test_word_parts_upper():
    tc = TypeContext()
    _create_root_types(tc)
    _create_all_word_parts(
        tc, [('foo', True), ("bar", True), ("fo", True), ("!", False)])
    tc.finalize_data()
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
    pointers = list(ast.depth_first_iter())
    assert ast == pointers[0].cur_node
    assert result.pointer_to_string(pointers[0]) == "FOO"
    assert word_part_o == pointers[1].cur_node
    assert result.pointer_to_string(pointers[1]) == "FOO"
    assert mod_type_choice == pointers[2].cur_node
    assert result.pointer_to_string(pointers[2]) == "FOO"
    assert mod_type_object == pointers[3].cur_node
    assert result.pointer_to_string(pointers[3]) == ""


def get_str_and_assert_same_part(result, pointer, node):
    assert pointer.cur_node == node
    return result.pointer_to_string(pointer)


def test_word_parts_2():
    tc = TypeContext()
    _create_root_types(tc)
    _create_all_word_parts(
        tc, [('foo', True), ("bar", True), ("fo", True), ("!", False)])
    tc.finalize_data()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("fooBarBaz", WORD_PART_TYPE_NAME, allow_partial_consume=True)
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
    pointers = list(ast.depth_first_iter())
    assert get_str_and_assert_same_part(result, pointers[1], word_part_o) == "fooBar"
    assert get_str_and_assert_same_part(result, pointers[2], mod_type_choice) == "foo"
    assert get_str_and_assert_same_part(result, pointers[3], mod_type_object) == ""
    assert get_str_and_assert_same_part(result, pointers[4], next_type_choice) == "Bar"
    assert get_str_and_assert_same_part(result, pointers[5], next_part_o) == "Bar"


def test_word_parts_3():
    tc = TypeContext()
    _create_root_types(tc)
    _create_all_word_parts(
        tc, [('f', True), ('ooo', True), ("bar", True), ("!", False)])
    tc.finalize_data()
    word_part_type = tc.get_type_by_name(WORD_PART_TYPE_NAME)
    parser = StringParser(tc)
    node, data = parser._parse_object_choice_node("fooo.bar", word_part_type.default_type_parser,
                                     word_part_type)
    assert data.parse_success
    assert data.remaining_string == ".bar"


def test_generic_word():
    context = TypeContext()
    loader.load_path("builtin_types/generic_parsers.ainix.yaml", context, up_search_limit=4)
    generic_strings.create_generic_strings(context)
    context.finalize_data()
    parser = StringParser(context)
    ast = parser.create_parse_tree("a", WORD_TYPE_NAME)
    generic_word_ob = ast.next_node_not_copy
    assert generic_word_ob.implementation.name == WORD_OBJ_NAME
    parts_arg = generic_word_ob.get_choice_node_for_arg("parts")
    parts_v = parts_arg.next_node_not_copy
    assert parts_v.implementation.name == "word_part_a"
    mod_type_choice = parts_v.get_choice_node_for_arg(WORD_PART_MODIFIER_ARG_NAME)
    mod_type_object = mod_type_choice.next_node_not_copy
    assert mod_type_object.implementation.name == MODIFIER_LOWER_NAME
    unparser = AstUnparser(context)
    result = unparser.to_string(ast)
    assert result.total_string == "a"

