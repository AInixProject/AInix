from types import GeneratorType
import pytest
from unittest.mock import MagicMock

from ainix_common.parsing.grammar_lang import *
from ainix_common.parsing.parse_primitives import ArgParseDelegation, ArgParseDelegationReturn


def test_parse():
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", "Foo")
    p_res = instance.parse_string("hello", MagicMock())
    assert isinstance(p_res, GeneratorType)
    first_del: ArgParseDelegation = next(p_res)
    assert first_del.string_to_parse == "hello"
    assert first_del.arg_name == "Foo"
    p_res.send(ArgParseDelegationReturn(True, ""))


def test_parse_2():
    instance = create_object_parser_from_grammar(MagicMock(), "FooParser", "Foo Bar")
    p_res = instance.parse_string("hello20", MagicMock())
    assert isinstance(p_res, GeneratorType)
    delegation: ArgParseDelegation = next(p_res)
    assert delegation.string_to_parse == "hello20"
    assert delegation.arg_name == "Foo"
    delegation = p_res.send(ArgParseDelegationReturn(True, "20"))
    assert delegation.string_to_parse == "20"
    assert delegation.arg_name == "Bar"
    p_res.send(ArgParseDelegationReturn(True, ""))
