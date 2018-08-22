import pytest
from parseast import *
import posix_parseing
from unittest.mock import MagicMock
import loader
from typecontext import TypeContext, AInixArgument, AInixObject


@pytest.fixture(scope="function")
def type_context():
    context = TypeContext()
    loader.load_path("../../builtin_types/command.ainix.yaml", context)
    return context


def test_object_choice_node_eq():
    mock_type = MagicMock()
    node1 = ObjectChoiceNode(mock_type, None)
    node2 = ObjectChoiceNode(mock_type, None)
    assert node1 == node2
    mock_object = MagicMock()
    mock_object.type = node1.type_to_choose
    node1.add_valid_choice(mock_object, 1)
    assert node1 != node2
    node2.add_valid_choice(mock_object, 0.5)
    assert node1 != node2
    node2.add_valid_choice(mock_object, 0.5)
    assert node1 == node2


def test_parse_object_choice1():
    mock_choice_node = MagicMock()
    mock_type_parser = MagicMock()
    mock_parse_result = MagicMock()
    mock_type_parser.parse_string.return_value = mock_parse_result
    pref_weight = 1.5
    result = StringParser._parse_object_choice(
        "foo", mock_choice_node, mock_type_parser, pref_weight)
    mock_type_parser.parse_string.assert_called_once_with("foo")
    # TODO: make this less of a kinda useless test...


def test_end_to_end_parse1(type_context):
    aArg = AInixArgument(type_context, "a", None, arg_data={"short_name": "a"})
    foo = AInixObject(
        type_context, "FooProgram", "Program",
        [aArg,
         AInixArgument(type_context, "b", None, arg_data={"short_name": "b"})],
        type_data={"invoke_name": "foo"})
    bar = AInixObject(
        type_context, "BarProgram", "Program",
        [AInixArgument(type_context, "b", None, arg_data={"short_name": "b"})],
        type_data={"invoke_name": "bar"})
    cmdSeqType = type_context.get_type_by_name("CommandSequence")
    weight = 2.0
    parser = StringParser(cmdSeqType)
    result = parser.create_parse_tree("foo -a", weight)
    assert "CommandSequenceObj" in result._valid_choices
    assert len(result._valid_choices) == 1
    c = result._valid_choices["CommandSequenceObj"]
    assert c.weight == weight
    # build expected tree
    expected = ObjectChoiceNode(cmdSeqType, None)
    seq_node = expected.add_valid_choice(
        type_context.get_object_by_name("CommandSequenceObj"), weight)
    prog_choice = seq_node.set_arg_present(seq_node.implementation.children[0])
    foo_object = prog_choice.add_valid_choice(foo, weight)
    v = foo_object.set_arg_present(aArg)
    assert v is None
    # compare
    print("result")
    print(result.dump_str())
    print("expected")
    print(expected.dump_str())
    assert result.dump_str() == expected.dump_str()
    assert result == expected
