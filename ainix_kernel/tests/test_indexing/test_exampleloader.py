from unittest.mock import MagicMock
from indexing.exampleloader import *
import io


def test_example():
    test_string = (
        "---\n"
        "defines:\n"
        "    - define_new: example_set\n"
        "      y_type: fooType\n"
        "      examples:\n"
        "          - x: Hello\n"
        "            y: Bonjour\n"
        "...\n"
    )
    f = io.StringIO(test_string)
    mock_index = MagicMock()
    load_yaml(f, mock_index)
    mock_index.add_many_to_many_default_weight.assert_called_once_with(
        ["Hello"], ["Bonjour"],
        mock_index.DEFAULT_X_TYPE, "fooType")


def test_example_2():
    test_string = (
        "---\n"
        "defines:\n"
        "    - define_new: example_set\n"
        "      y_type: fooType\n"
        "      examples:\n"
        "          - x: Hello\n"
        "            y:\n"
        "               - Bonjour\n"
        "               - Salut\n"
        "...\n"
    )
    f = io.StringIO(test_string)
    mock_index = MagicMock()
    load_yaml(f, mock_index)
    mock_index.add_many_to_many_default_weight.assert_called_once_with(
        ["Hello"], ["Bonjour", "Salut"],
        mock_index.DEFAULT_X_TYPE, "fooType")
