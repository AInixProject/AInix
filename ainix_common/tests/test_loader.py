import pytest
from loader import *
import io
from unittest.mock import MagicMock


def test_single_type():
    test_string = (
       "---\n"
       "defines:\n"
       "    - define_new: type\n"
       "      name: foo\n"
       "      default_type_parser: foo_type_p\n"
       "      default_object_parser: foo_obj_p\n"
       "...\n"
    )
    print(test_string)
    f = io.StringIO(test_string)
    mock_type_graph = MagicMock()
    load_yaml(f, mock_type_graph)
    mock_type_graph.create_type.assert_called_once_with(
        "foo", default_type_parser="foo_type_p",
        default_object_parser="foo_obj_p",
        allowed_attributes=None)
