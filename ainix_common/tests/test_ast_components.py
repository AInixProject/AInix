from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode, ObjectNode
from ainix_common.parsing.typecontext import TypeContext, AInixType, AInixArgument, AInixObject
from unittest.mock import MagicMock
import pytest


def test_parse_set_optional():
    """Manually build up an ast, and make sure the ast set is representing
    optional args correctly."""
    tc = TypeContext()
    foo_type = AInixType(tc, "FooType")
    test_arg = AInixArgument(tc, "test_arg", "FooType")
    foo_ob = AInixObject(tc, "foo_ob", "FooType", [test_arg])
    ast_set = AstObjectChoiceSet(foo_type, None)
    ast = ObjectChoiceNode(foo_type)
    next_o = ObjectNode(foo_ob)
    ast.set_choice(next_o)
    arg_choice_node = ObjectChoiceNode(test_arg.present_choice_type)
    arg_choice_node.set_choice(ObjectNode(test_arg.not_present_object))
    next_o.set_arg_value("test_arg", arg_choice_node)

    ast.freeze()
    ast_set.add(ast, True, 1, 1)

    data = ast_set._impl_name_to_data["foo_ob"].next_node.get_arg_set_data(
        next_o.as_childless_node())
    assert data is not None
    assert data.arg_to_choice_set["test_arg"].type_to_choose_name == \
           test_arg.present_choice_type.name


def test_objectchoice_node_copy_frozen():
    mock_choice = MagicMock()
    instance = ObjectChoiceNode(MagicMock(), mock_choice)
    clone, path = instance.path_clone()
    assert id(clone) == id(instance)
    assert path is None


def test_objectchoice_node_copy_frozen_with_child():
    mock_choice = MagicMock()
    mock_choice.path_copy.returns = mock_choice
    instance = ObjectChoiceNode(MagicMock(), mock_choice)
    clone, path = instance.path_clone()
    assert id(clone) == id(instance)
    assert clone.choice == mock_choice
    assert path is None


def test_objectchoice_node_copy_frozen_on_path():
    mock_choice = MagicMock()
    instance = ObjectChoiceNode(MagicMock(), mock_choice)
    clone, path = instance.path_clone([instance])
    assert id(clone) != id(instance)
    assert clone.choice is None
    assert not clone.is_frozen
    assert path == [clone]


def test_objectchoice_node_copy_frozen_on_paths():
    mock_choice = MagicMock()
    mock_copy = MagicMock()
    mock_choice.path_clone.return_value = (mock_copy, [mock_copy])
    instance = ObjectChoiceNode(MagicMock(), mock_choice)
    clone, path = instance.path_clone([instance, mock_copy])
    assert id(clone) != id(instance)
    assert clone.choice == mock_copy
    assert not clone.is_frozen
    assert path[0] == clone
    assert path[1] == mock_copy


def test_objectnode_copy_simple():
    """Copy with no children"""
    tc = TypeContext()
    AInixType(tc, "footype")
    foo_object = AInixObject(tc, "foo_object", "footype")
    instance = ObjectNode(foo_object)
    # Unfrozen
    clone, path = instance.path_clone()
    assert id(clone) != id(instance)
    assert clone.implementation == foo_object
    assert instance == clone
    assert path is None
    # Frozen
    instance.freeze()
    clone, path = instance.path_clone()
    assert id(clone) == id(instance)
    assert clone == instance
    assert path is None
    # Frozen but on unfreeze path
    clone, path = instance.path_clone([instance])
    assert id(clone) != id(instance)
    assert instance == clone
    assert clone.implementation == foo_object


#def test_objectnode_copy_with_child():
#    """Copy with no children unfrozen"""
#    tc = TypeContext()
#    AInixType(tc, "footype")
#    foo_object = AInixObject(tc, "foo_object", "footype")
#    instance = ObjectNode(foo_object)
#    clone, path = instance.path_clone()





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
