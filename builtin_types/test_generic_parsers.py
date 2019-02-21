import pytest

from ainix_common.parsing.stringparser import StringParser
from ainix_common.parsing.typecontext import TypeContext, AInixType
from builtin_types.generic_parsers import *
from ainix_common.parsing import loader
from unittest.mock import MagicMock


def test_max_munch():
    tc = TypeContext()
    loader.load_path("builtin_types/generic_parsers.ainix.yaml", tc, up_search_limit=3)
    foo_type = "MMTestType"
    AInixType(tc, foo_type, default_type_parser_name="max_munch_type_parser")

    def make_mock_with_parse_rep(representation: str):
        loader._load_object({
            "name": representation,
            "type": foo_type,
            "preferred_object_parser": {
                "grammar": f'"{representation}"'
            }
        }, tc, "foopathsdf")
        assert tc.get_object_by_name(representation).preferred_object_parser_name is not None
    objects = [make_mock_with_parse_rep(rep)
               for rep in ("fo", "bar", "f", "foo", "foot", 'baz')]

    parser = StringParser(tc)
    ast = parser.create_parse_tree("foobar", foo_type, allow_partial_consume=True)
    assert ast.next_node_not_copy.implementation.name == "foo"

    #with pytest.raises(parse_primitives.AInixParseError):
    #    parse_str = "moo"
    #    result = parse_primitives.TypeParserResult(MagicMock, parse_str)
    #    max_munch_type_parser(mock_run, parse_str, result)


#def test_max_munch_empty_string():
#    def make_mock_with_parse_rep(representation: str):
#        out = MagicMock()
#        out.type_data = {MAX_MUNCH_LOOKUP_KEY: representation}
#        return out
#    mock_run = MagicMock()
#    mock_types = [make_mock_with_parse_rep(rep)
#                  for rep in ("fo", "bar", "")]
#    mock_run.all_type_implementations = mock_types
#    parse_str = "moo"
#    result = parse_primitives.TypeParserResult(MagicMock, parse_str)
#    max_munch_type_parser(mock_run, parse_str, result)
#    assert result.get_implementation() == mock_types[2]
#    assert result.get_next_string() == "moo"


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
    run = parse_primitives.ObjectParserRun(mock_object, test_string)
    result = parse_primitives.ObjectParserResult(mock_object, test_string)
    regex_group_object_parser(run, test_string, result)
    a = result.get_arg_present("a")
    assert a.slice == (0, 1)
    assert a.slice_string == "1"
    b = result.get_arg_present("b")
    assert b.slice == (1, 3)
    assert b.slice_string == "23"
    c = result.get_arg_present("c")
    assert c is None

