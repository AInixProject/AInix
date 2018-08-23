import pytest
from parse_primitives import *
from unittest.mock import MagicMock
from typecontext import AInixObject, AInixArgument

def test_add_arg():
    mock_context = MagicMock()
    obj = AInixObject(mock_context, "FooObj", "footype",
                      [AInixArgument(mock_context, "FooArg", None)])
    test_string = "test string"
    result = ObjectParserResult(obj, test_string)

    # Invalid arg
    with pytest.raises(Exception):
        result.set_arg_present("badname", 0, 4)
    # Valid arg
    result.set_arg_present("FooArg", 0, 4)
    assert result.get_arg_present("badname") is None
    data = result.get_arg_present("FooArg")
    assert data is not None
    assert data.slice == (0, 4)
    assert data.slice_string == test_string[0:4]
