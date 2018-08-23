from generic_parsers import *
from unittest.mock import MagicMock
import typecontext
import parse_primitives

def test_max_munch():
    mock_parser = MagicMock()
    def make_mock_with_parse_rep(representation: str):
        out = MagicMock()
        out.type_data = {"ParseRepresentation": representation}
        return out
    mock_types = [make_mock_with_parse_rep(rep)
                  for rep in ("fo", "bar", "f", "foo", "foot", 'baz')]
    mock_parser.type_implementations = mock_types
    parse_str = "foobar"
    result = parse_primitives.TypeParserResult(MagicMock, parse_str)
    max_munch_type_parser(mock_parser, parse_str, result)
    assert result.get_implementation() == mock_types[3]
    assert result.get_next_string() == "foo"





