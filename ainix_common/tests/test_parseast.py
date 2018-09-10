import pytest
from parseast import *
from unittest.mock import MagicMock
import loader
from typecontext import TypeContext, AInixArgument, AInixObject

BUILTIN_TYPES_PATH = "../../builtin_types"

@pytest.fixture(scope="function")
def type_context():
    context = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/command.ainix.yaml", context)
    context.fill_default_parsers()
    return context


@pytest.fixture(scope="function")
def numbers_type_context():
    type_context = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/generic_parsers.ainix.yaml", type_context)
    loader.load_path(f"{BUILTIN_TYPES_PATH}/numbers.ainix.yaml", type_context)
    type_context.fill_default_parsers()
    return type_context


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
    parser = StringParser(type_context)
    result = parser.create_parse_tree("foo -a", cmdSeqType.name)
    assert result.type_to_choose == cmdSeqType
    assert result.choice.implementation == type_context.get_object_by_name("CommandSequenceObj")
    compoundOp: ObjectChoiceNode = result.choice._arg_name_to_node['CompoundOp']
    assert compoundOp.choice.implementation.name.endswith("NOTPRESENT")
    programArg: ObjectChoiceNode = result.choice._arg_name_to_node['ProgramArg']
    assert programArg.type_to_choose == type_context.get_type_by_name("Program")
    assert programArg.choice.implementation == foo
    a_choice: ObjectChoiceNode = programArg.choice._arg_name_to_node[aArg.name]
    assert not a_choice.choice.implementation.name.endswith("NOTPRESENT")
    b_choice: ObjectChoiceNode = programArg.choice._arg_name_to_node[bArg.name]
    assert b_choice.choice.implementation.name.endswith("NOTPRESENT")


@pytest.fixture(scope="function")
def numbers_ast_set(numbers_type_context) -> AstObjectChoiceSet:
    root_type_name = "Number"
    root_type = numbers_type_context.get_type_by_name(root_type_name)
    choice_set = AstObjectChoiceSet(root_type, None)
    return choice_set


def test_parse_set_1(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast = parser.create_parse_tree("5", root_type_name)
    numbers_ast_set.add(ast, True, 1, 1)
    assert numbers_ast_set.is_node_known_valid(ast)
    assert not numbers_ast_set.is_node_known_valid(
        parser.create_parse_tree("9", root_type_name))
    assert not numbers_ast_set.is_node_known_valid(
        parser.create_parse_tree("-5", root_type_name))


def test_parse_set_freeze(numbers_type_context, numbers_ast_set):
    root_type_name = "Number"
    parser = StringParser(numbers_type_context)
    ast = parser.create_parse_tree("5", root_type_name)
    numbers_ast_set.add(ast, True, 1, 1)
    numbers_ast_set.freeze()
    with pytest.raises(ValueError):
        numbers_ast_set.add(ast, True, 1, 1)
    real_set = {numbers_ast_set}
    assert numbers_ast_set in real_set


def test_parse_set_2(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("-5", root_type_name)
    print(ast_1.dump_str())
    numbers_ast_set.add(ast_1, True, 1, 1)
    print("---")
    assert numbers_ast_set.is_node_known_valid(ast_1)


def test_parse_set_3(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("-5", root_type_name)
    ast_3 = parser.create_parse_tree("50", root_type_name)
    ast_4 = parser.create_parse_tree("6", root_type_name)
    numbers_ast_set.add(ast_1, True, 1, 1)
    numbers_ast_set.add(ast_2, True, 1, 1)
    numbers_ast_set.add(ast_3, True, 1, 0.2)
    numbers_ast_set.add(ast_4, True, 1, 1)
    assert numbers_ast_set.is_node_known_valid(ast_1)
    assert numbers_ast_set.is_node_known_valid(ast_2)
    assert numbers_ast_set.is_node_known_valid(ast_3)
    assert numbers_ast_set.is_node_known_valid(ast_4)


def test_parse_set_4(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    numbers_ast_set.add(ast_1, False, 1, 1)
    assert not numbers_ast_set.is_node_known_valid(ast_1)


def test_parse_set_5(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("-5", root_type_name)
    numbers_ast_set.add(ast_1, True, 1, 1)
    numbers_ast_set.add(ast_2, False, 1, 1)
    assert numbers_ast_set.is_node_known_valid(ast_1)
    assert not numbers_ast_set.is_node_known_valid(ast_2)


def test_parse_set_6(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("50", root_type_name)
    numbers_ast_set.add(ast_1, True, 1, 1)
    print("---")
    numbers_ast_set.add(ast_2, False, 1, 0.3)
    assert numbers_ast_set.is_node_known_valid(ast_1)
    assert not numbers_ast_set.is_node_known_valid(ast_2)


def test_parse_set_7(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("50", root_type_name)
    numbers_ast_set.add(ast_1, False, 1, 1)
    numbers_ast_set.add(ast_2, True, 1, 0.3)
    assert not numbers_ast_set.is_node_known_valid(ast_1)
    assert numbers_ast_set.is_node_known_valid(ast_2)


def test_parse_set_8(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("50", root_type_name)
    ast_3 = parser.create_parse_tree("500", root_type_name)
    numbers_ast_set.add(ast_1, True, 1, 1)
    numbers_ast_set.add(ast_2, False, 1, 0.3)
    numbers_ast_set.add(ast_3, True, 1, 1)
    assert numbers_ast_set.is_node_known_valid(ast_1)
    assert not numbers_ast_set.is_node_known_valid(ast_2)
    assert numbers_ast_set.is_node_known_valid(ast_3)


#def test_parse_set_weights_1(numbers_type_context, numbers_ast_set):
#    parser = StringParser(numbers_type_context)
#    root_type_name = "Number"
#    ast_1 = parser.create_parse_tree("5", root_type_name)
#    ast_2 = parser.create_parse_tree("50", root_type_name)
#    ast_3 = parser.create_parse_tree("500", root_type_name)
#    numbers_ast_set.add(ast_1, True, 1, 1)
#    numbers_ast_set.add(ast_2, True, 1, 0.3)
#    numbers_ast_set.add(ast_3, True, 1, 1)
#    assert numbers_ast_set.is_node_known_valid(ast_1)
#    assert not numbers_ast_set.is_node_known_valid(ast_2)
#    assert numbers_ast_set.is_node_known_valid(ast_3)
