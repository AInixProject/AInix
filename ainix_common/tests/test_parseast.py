import pytest
from parseast import *
from unittest.mock import MagicMock
import loader
from typecontext import TypeContext, AInixArgument, AInixObject


@pytest.fixture(scope="function")
def type_context():
    context = TypeContext()
    loader.load_path("../../builtin_types/command.ainix.yaml", context)
    context.fill_default_parsers()
    return context


# broken when weight thing changed
#def test_object_choice_node_eq():
#    mock_type = MagicMock()
#    node1 = ObjectChoiceNode(mock_type, None)
#    node2 = ObjectChoiceNode(mock_type, None)
#    assert node1 == node2
#    mock_object = MagicMock()
#    mock_object.type = node1._type_to_choose
#    node1.add_valid_choice(mock_object, 1)
#    assert node1 != node2
#    node2.add_valid_choice(mock_object, 0.5)
#    assert node1 != node2
#    node2.add_valid_choice(mock_object, 0.5)
#    assert node1 == node2

def test_end_to_end_parse1(type_context):
    aArg = AInixArgument(type_context, "a", None, arg_data={"short_name": "a"})
    bArg = AInixArgument(type_context, "b", None, arg_data={"short_name": "b"})
    foo = AInixObject(
        type_context, "FooProgram", "Program",
        [aArg, bArg],
        type_data={"invoke_name": "foo"})
    bar = AInixObject(
        type_context, "BarProgram", "Program",
        [AInixArgument(type_context, "b", None, arg_data={"short_name": "b"})],
        type_data={"invoke_name": "bar"})
    cmdSeqType = type_context.get_type_by_name("CommandSequence")
    parser = StringParser(cmdSeqType)
    result = parser.create_parse_tree("foo -a")
    assert result.type_to_choose == cmdSeqType
    assert result.choice.implementation == type_context.get_object_by_name("CommandSequenceObj")
    compoundOp: ArgPresentChoiceNode = result.choice.arg_name_to_node['CompoundOp']
    assert not compoundOp.is_present
    programArg: ObjectChoiceNode = result.choice.arg_name_to_node['ProgramArg']
    assert programArg.type_to_choose == type_context.get_type_by_name("Program")
    assert programArg.choice.implementation == foo
    assert programArg.choice.arg_name_to_node == {
        aArg.name : ArgPresentChoiceNode(aArg, True, None),
        bArg.name: ArgPresentChoiceNode(bArg, False, None)
    }
