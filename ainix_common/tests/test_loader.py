import pytest
from loader import *
import io
from unittest.mock import MagicMock, patch


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
    mock_type_context = MagicMock()

    def mock_register(new_type: typecontext.AInixType):
        assert new_type.name == "foo"
        assert new_type.default_type_parser_name == "foo_type_p"
        assert new_type.default_object_parser_name == "foo_obj_p"
    mock_type_context.register_type = mock_register
    load_yaml(f, mock_type_context, "")


def test_single_obj():
    test_string = (
        "---\n"
        "defines:\n"
        "    - define_new: object\n"
        "      name: foo\n"
        "      type: foo_t\n"
        "      children:\n"
        "          - name: test_arg\n"
        "            type: foo_type\n"
        "          - name: test_arg_2\n"
        "            type: bar_type\n"
        "            required: true\n"
        "...\n"
    )
    print(test_string)
    f = io.StringIO(test_string)
    mock_type_context = MagicMock()

    def mock_register(new_object: typecontext.AInixObject):
        assert new_object.name == "foo"
        assert new_object.type_name == "foo_t"
        arg0 = new_object.children[0]
        assert arg0.name == "test_arg"
        assert arg0.type_name == "foo_type"
        assert not arg0.required
        arg1 = new_object.children[1]
        assert arg1.name == "test_arg_2"
        assert arg1.type_name == "bar_type"
        assert arg1.required
    load_yaml(f, mock_type_context, "")
    mock_type_context.register_object = mock_register

