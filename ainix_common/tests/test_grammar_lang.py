from types import GeneratorType
import pytest
from unittest.mock import MagicMock

from ainix_common.parsing.grammar_lang import *
from ainix_common.parsing.parse_primitives import ArgParseDelegation, \
    ParseDelegationReturnMetadata, StringProblemParseError, ObjectNodeArgMap, \
    ArgToStringDelegation
from ainix_common.parsing.typecontext import AInixArgument


def send_result(gen, send_value):
    """Get the result from a generator when expected to return next."""
    try:
        gen.send(send_value)
    except StopIteration as stp:
        return stp.value
    raise ValueError("Expected it to stop")


@pytest.fixture()
def mobject():
    """Makes a nice mock object that returns if an argument is ever requested
    it magically makes an argument with that name"""
    out = MagicMock()

    def magic_get_arg_by_name(name):
        new_arg = AInixArgument(MagicMock(), name, None)
        return new_arg
    out.get_arg_by_name = magic_get_arg_by_name
    return out


def test_parse(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", "Foo")
    p_res = instance.parse_string("hello", mobject)
    assert isinstance(p_res, GeneratorType)
    first_del: ArgParseDelegation = next(p_res)
    assert first_del.string_to_parse == "hello"
    assert first_del.arg.name == "Foo"
    assert first_del.slice_to_parse == (0, 5)
    response = first_del.next_from_substring("")
    assert response.original_start_offset == 0
    try:
        p_res.send(response)
    except StopIteration as stp:
        result = stp.value
    arg_data = result.get_arg_present("Foo")
    assert arg_data is not None
    assert arg_data.set_from_delegation == response
    assert arg_data.slice == (0, 5)
    assert result.remaining_start_i == 5


def test_parse_2(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", "Foo Bar")
    p_res = instance.parse_string("hello20blah", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello20blah"
    assert delegation.arg.name == "Foo"
    response = delegation.next_from_substring("20blah")
    delegation = p_res.send(response)
    assert delegation.string_to_parse == "20blah"
    assert delegation.arg.name == "Bar"
    assert delegation.slice_to_parse == (5, 11)
    response = delegation.next_from_substring("blah")
    assert response.original_start_offset == 5
    assert response.remaining_right_starti == 2
    result = send_result(p_res, response)
    foo_data = result.get_arg_present("Foo")
    assert foo_data is not None
    assert foo_data.slice == (0, 5)
    assert foo_data.slice_string == "hello"
    bar_data = result.get_arg_present("Bar")
    assert bar_data is not None
    assert bar_data.slice_string == "20"
    assert bar_data.slice == (5, 7)
    assert result.remaining_start_i == 7


def test_parse_with_err(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", "Foo Bar")
    p_res = instance.parse_string("hello20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello20"
    assert delegation.arg.name == "Foo"
    with pytest.raises(UnparseableObjectError):
        delegation = p_res.send(ParseDelegationReturnMetadata(False, "hello20", 0,
                                                              delegation.arg, None))


def test_parse_str_litteral(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'Foo "-" Bar')
    p_res = instance.parse_string("hello-20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello-20"
    assert delegation.arg.name == "Foo"
    delegation = p_res.send(delegation.next_from_substring("-20"))
    assert delegation.string_to_parse == "20"
    assert delegation.arg.name == "Bar"
    result = send_result(p_res, delegation.next_from_substring(""))
    bar_data = result.get_arg_present("Bar")
    assert bar_data.slice == (6, 8)


def test_parse_str_litteral_fail(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'Foo "-" Bar')
    p_res = instance.parse_string("hello=20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    with pytest.raises(StringProblemParseError):
        delegation = p_res.send(delegation.next_from_substring("=20"))


def test_parse_optional(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", "Foo Bar?")
    p_res = instance.parse_string("hello20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello20"
    assert delegation.arg.name == "Foo"
    delegation = p_res.send(delegation.next_from_substring("20"))
    assert delegation.string_to_parse == "20"
    assert delegation.arg.name == "Bar"
    try:
        p_res.send(ParseDelegationReturnMetadata.make_failing())
    except StopIteration as stp:
        result = stp.value
    assert result.get_arg_present("Foo") is not None
    assert result.get_arg_present("Bar") is None


def test_parse_litteral_after(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'Foo "There"')
    p_res = instance.parse_string("helloThere", mobject)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "helloThere"
    assert delegation.arg.name == "Foo"
    try:
        p_res.send(delegation.next_from_substring("There"))
    except StopIteration as stp:
        result = stp.value
    assert result.get_arg_present("Foo") is not None
    # Now try something which doesn't have the litteral
    p_res = instance.parse_string("helloHere", mobject)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "helloHere"
    with pytest.raises(StringProblemParseError):
        p_res.send(delegation.next_from_substring("Here"))


def test_parse_litteral_paren(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'Foo ("-" Bar)?')
    p_res = instance.parse_string("hello-20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello-20"
    assert delegation.arg.name == "Foo"
    delegation = p_res.send(delegation.next_from_substring("-20"))
    assert delegation.string_to_parse == "20"
    assert delegation.arg.name == "Bar"
    result = send_result(p_res, delegation.next_from_substring(""))
    bar_data = result.get_arg_present("Bar")
    assert bar_data.slice == (6, 8)


def test_parse_litteral_paren_fail(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'Foo ("-" Bar)?')
    p_res = instance.parse_string("hello>20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello>20"
    assert delegation.arg.name == "Foo"
    try:
        p_res.send(delegation.next_from_substring(">20"))
    except StopIteration as stp:
        result = stp.value
    assert result.get_arg_present("Foo") is not None
    assert result.get_arg_present("Bar") is None


def test_parse_litteralafter_paren(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'Foo (Bar "!")?')
    p_res = instance.parse_string("hello20!", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello20!"
    assert delegation.arg.name == "Foo"
    delegation = p_res.send(delegation.next_from_substring("20!"))
    assert delegation.string_to_parse == "20!"
    assert delegation.arg.name == "Bar"
    result = send_result(p_res, delegation.next_from_substring("!"))
    assert result.remaining_start_i == 8


def test_parse_litteralafter_paren_fail(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'Foo (Bar "!")?')
    p_res = instance.parse_string("hello20.", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello20."
    delegation = p_res.send(delegation.next_from_substring("20."))
    assert delegation.string_to_parse == "20."
    assert delegation.arg.name == "Bar"
    try:
        p_res.send(delegation.next_from_substring("."))
    except StopIteration as stp:
        result = stp.value
    assert result.get_arg_present("Foo") is not None
    assert result.get_arg_present("Bar") is None
    assert result.remaining_start_i == 5


def test_unparse_juststr(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", '"Hello"')
    args = ObjectNodeArgMap(mobject, {})
    result = instance.to_string(args)
    assert result.unparse_seq == ["Hello",]


def test_unparse_arg(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'FooArg')
    foo_arg = mobject.get_arg_by_name("FooArg")
    args = ObjectNodeArgMap(mobject, {"FooArg": True})
    result = instance.to_string(args)
    assert result.unparse_seq == [ArgToStringDelegation(foo_arg)]


def test_unparse_fail(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'FooArg')
    args = ObjectNodeArgMap(mobject, {"FooArg": False})
    with pytest.raises(UnparseError):
        instance.to_string(args)


def test_unparse_str_and_arg(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", '"Hello" FooArg')
    foo_arg = mobject.get_arg_by_name("FooArg")
    args = ObjectNodeArgMap(mobject, {"FooArg": True})
    result = instance.to_string(args)
    assert result.unparse_seq == ["Hello", ArgToStringDelegation(foo_arg)]


def test_unparse_suffix(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", '"Hey" FooArg?')
    args = ObjectNodeArgMap(mobject, {"FooArg": False})
    result = instance.to_string(args)
    assert result.unparse_seq == ["Hey"]
