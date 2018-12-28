from ainix_common.parsing.parse_primitives import TypeParserResult
from ainix_common.tests.testutils import send_result
from ainix_kernel.specialtypes.generic_strings import *
from ainix_kernel.specialtypes.generic_strings import \
    _get_all_symbols, _create_root_types, _create_all_word_parts, _name_for_word_part


def test_word_parts():
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
