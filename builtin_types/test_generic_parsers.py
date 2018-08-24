import pytest
from generic_parsers import *
from unittest.mock import MagicMock
import typecontext
import parse_primitives


def test_max_munch():
    def make_mock_with_parse_rep(representation: str):
        out = MagicMock()
        out.type_data = {MAX_MUNCH_LOOKUP_KEY: representation}
        return out
    mock_run = MagicMock()
    mock_types = [make_mock_with_parse_rep(rep)
                  for rep in ("fo", "bar", "f", "foo", "foot", 'baz')]
    mock_run.all_type_implementations = mock_types
    parse_str = "foobar"
    result = parse_primitives.TypeParserResult(MagicMock, parse_str)
    max_munch_type_parser(mock_run, parse_str, result)
    assert result.get_implementation() == mock_types[3]
    assert result.get_next_string() == "foo"

    with pytest.raises(parse_primitives.AInixParseError):
        parse_str = "moo"
        result = parse_primitives.TypeParserResult(MagicMock, parse_str)
        max_munch_type_parser(mock_run, parse_str, result)


def test_regex_group_parser():
    def make_mock_arg_with_regex_data(name, data: str):
        out = MagicMock()
        out.name = name
        out.arg_data = {REGEX_GROUP_LOOKUP_KEY: data}
        out.required = False
        return out
    mock_parser = MagicMock()
    mock_args = [make_mock_arg_with_regex_data(name, regex)
                 for name, regex in
                 (("a", r'([0-9]).*'),
                  ("b", r"[0-9]([0-9]+).*"),
                  ("c", r"(.*)bar"))]
    mock_object = MagicMock()
    mock_object.children = mock_args
    test_string = "123foo"
    result = parse_primitives.ObjectParserResult(mock_object, test_string)
    regex_group_object_parser(mock_parser, mock_object, test_string, result)
    a = result.get_arg_present("a")
    assert a.slice == (0, 1)
    assert a.slice_string == "1"
    b = result.get_arg_present("b")
    assert b.slice == (1, 3)
    assert b.slice_string == "23"
    c = result.get_arg_present("c")
    assert c is None

