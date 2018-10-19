import pytest
from ainix_common.parsing.parse_primitives import *
from ainix_common.parsing.typecontext import AInixObject, AInixArgument
from unittest.mock import MagicMock

def test_add_arg():
    mock_context = MagicMock()
    obj = AInixObject(mock_context, "FooObj", "footype",
                      [AInixArgument(mock_context, "FooArg", None)])
    test_string = "test string"
    result = ObjectParserResult(obj, test_string)

    # Invalid arg
    with pytest.raises(Exception):
        result.set_arg_present("badname", 0, 4)
    with pytest.raises(Exception):
        result.get_arg_present("badname")
    # Valid arg
    result.set_arg_present("FooArg", 0, 4)
    data = result.get_arg_present("FooArg")
    assert data is not None
    assert data.slice == (0, 4)
    assert data.slice_string == test_string[0:4]
    assert result.remaining_start_i == 4
