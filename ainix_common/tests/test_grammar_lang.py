from types import GeneratorType
import pytest
from unittest.mock import MagicMock

from ainix_common.parsing.grammar_lang import *
from ainix_common.parsing.parse_primitives import ArgParseDelegation, \
    ParseDelegationReturnMetadata, StringProblemParseError
from ainix_common.parsing.typecontext import AInixArgument


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
    response = first_del.next_from_substring("")
    try:
        p_res.send(response)
    except StopIteration as stp:
        result = stp.value
    arg_data = result.get_arg_present("Foo")
    assert arg_data is not None
    assert arg_data.set_from_delegation == response



def test_parse_2(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", "Foo Bar")
    p_res = instance.parse_string("hello20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello20"
    assert delegation.arg.name == "Foo"
    delegation = p_res.send(delegation.next_from_substring("20"))
    assert delegation.string_to_parse == "20"
    assert delegation.arg.name == "Bar"
    with pytest.raises(StopIteration):
        p_res.send(delegation.next_from_substring(""))


def test_parse_with_err(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", "Foo Bar")
    p_res = instance.parse_string("hello20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello20"
    assert delegation.arg.name == "Foo"
    with pytest.raises(UnparseableObjectError):
        delegation = p_res.send(ParseDelegationReturnMetadata(False, "hello20",
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
    with pytest.raises(StopIteration):
        p_res.send(delegation.next_from_substring(""))


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
        p_res.send(ParseDelegationReturnMetadata(False, "hell20", delegation.arg, None, "Just no"))
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
    with pytest.raises(StopIteration):
        p_res.send(delegation.next_from_substring(""))


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


