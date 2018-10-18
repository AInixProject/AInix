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
    p_res.send(first_del.next_from_substring(""))


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
    p_res.send(delegation.next_from_substring(""))


def test_parse_str_litteral_fail(mobject):
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", 'Foo "-" Bar')
    p_res = instance.parse_string("hello=20", mobject)
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    with pytest.raises(StringProblemParseError):
        delegation = p_res.send(delegation.next_from_substring("=20"))
