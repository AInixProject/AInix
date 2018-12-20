from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode, ObjectNode, \
    AstIterPointer
from ainix_common.parsing.typecontext import TypeContext, AInixType, AInixArgument, AInixObject, \
    OPTIONAL_ARGUMENT_NEXT_ARG_NAME
from unittest.mock import MagicMock
import pytest

from ainix_common.tests.toy_contexts import get_toy_strings_context


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
    clone, leaf_pointer = instance.path_clone()
    assert id(clone) == id(instance)
    assert leaf_pointer is None


def test_objectchoice_node_copy_frozen_with_child():
    mock_choice = MagicMock()
    mock_choice.path_copy.returns = mock_choice
    instance = ObjectChoiceNode(MagicMock(), mock_choice)
    clone, leaf_pointer = instance.path_clone()
    assert id(clone) == id(instance)
    assert clone.choice == mock_choice
    assert leaf_pointer is None


def test_objectchoice_node_copy_frozen_on_path():
    mock_choice = MagicMock()
    mock_copy = MagicMock()
    mock_choice.path_clone.return_value = (mock_copy, [mock_copy])
    instance = ObjectChoiceNode(MagicMock(), mock_choice)
    clone, leaf_pointer = instance.path_clone([instance])
    assert id(clone) != id(instance)
    assert clone.choice is None
    assert not clone.is_frozen
    assert leaf_pointer == AstIterPointer(clone, None, None)


def test_objectchoice_node_copy_frozen_on_paths():
    mock_choice = MagicMock()
    mock_copy = MagicMock()
    instance = ObjectChoiceNode(MagicMock(), mock_choice)
    last_pointer = AstIterPointer(mock_choice, AstIterPointer(instance, None, None), 0)
    mock_choice.path_clone.return_value = (mock_copy, last_pointer)
    clone, leaf_pointer = instance.path_clone([instance, mock_copy])
    assert id(clone) != id(instance)
    assert clone.choice == mock_copy
    assert not clone.is_frozen
    a = leaf_pointer.get_nodes_to_here()[0]
    assert isinstance(a, ObjectChoiceNode)
    assert a._type_to_choose == clone._type_to_choose
    assert leaf_pointer.cur_node == mock_choice
    assert leaf_pointer.parent_child_ind == 0
    assert leaf_pointer.parent.cur_node._type_to_choose == clone._type_to_choose


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
    assert clone.is_frozen
    # Frozen but on unfreeze path
    clone, path = instance.path_clone([instance])
    assert id(clone) != id(instance)
    assert instance == clone
    assert clone.implementation == foo_object
    assert not clone.is_frozen


def test_objectnode_copy_simple_with_arg():
    """Copy with an arg"""
    # Establish types
    tc = TypeContext()
    AInixType(tc, "footype")
    bartype = AInixType(tc, "bartype")
    arg1 = AInixArgument(tc, "arg1", "bartype", required=True)
    foo_object = AInixObject(tc, "foo_object", "footype", [arg1])
    bar_object = AInixObject(tc, "bar_object", "bartype")
    # Make an ast
    arg_choice = ObjectChoiceNode(bartype)
    ob_chosen = ObjectNode(bar_object)
    arg_choice.set_choice(ob_chosen)
    instance = ObjectNode(foo_object)
    instance.set_arg_value("arg1", arg_choice)
    # Do the tests:
    # Unfrozen
    clone, leaf = instance.path_clone()
    assert id(clone) != id(instance)
    assert leaf is None
    # with a path
    clone, leaf = instance.path_clone([instance])
    assert id(clone) != id(instance)
    assert leaf.cur_node == clone
    assert leaf.parent is None
    # with a deep path
    clone, leaf = instance.path_clone([instance, arg_choice])
    assert id(clone) != id(instance)
    assert leaf.cur_node.choice is None
    # with a deeper path
    clone, leaf = instance.path_clone([instance, arg_choice, ob_chosen])
    assert id(clone) != id(instance)
    assert clone == instance
    assert leaf.cur_node == ob_chosen
    assert leaf.parent.cur_node.choice == ob_chosen


def test_objectnode_copy_with_child():
    """Copy with an arg"""
    # Establish types
    tc = TypeContext()
    AInixType(tc, "footype")
    bartype = AInixType(tc, "bartype")
    arg1 = AInixArgument(tc, "arg1", "bartype")
    foo_object = AInixObject(tc, "foo_object", "footype", [arg1])
    bar_object = AInixObject(tc, "bar_object", "bartype")
    # Make an ast
    fin_choice = ObjectNode(bar_object)
    is_pres = ObjectChoiceNode(bartype)
    is_pres.set_choice(fin_choice)
    arg_node = ObjectNode(arg1.is_present_object)
    arg_node.set_arg_value(OPTIONAL_ARGUMENT_NEXT_ARG_NAME, is_pres)
    is_pres_top = ObjectChoiceNode(arg1.present_choice_type)
    is_pres_top.set_choice(arg_node)
    instance = ObjectNode(foo_object)
    instance.set_arg_value("arg1", is_pres_top)
    # Do the tests:
    # Unfrozen
    clone, leaf = instance.path_clone()
    assert id(clone) != id(instance)
    assert leaf is None
    # Freeze part
    is_pres_top.freeze()
    clone, leaf = instance.path_clone()
    assert id(clone) != id(instance)
    assert not clone.is_frozen
    assert clone == instance
    assert id(clone.get_choice_node_for_arg("arg1")) == id(is_pres_top)
    # Freeze all
    instance.freeze()
    clone, leaf = instance.path_clone()
    assert id(clone) == id(instance)
    assert clone == instance
    assert id(clone.get_choice_node_for_arg("arg1")) == id(is_pres_top)
    # Full unfreeze path
    clone, leaf = instance.path_clone([instance, is_pres_top, arg_node, is_pres, fin_choice])
    assert id(clone) != id(instance)
    assert not clone.is_frozen
    assert clone == instance
    assert leaf.get_nodes_to_here() == [instance, is_pres_top, arg_node, is_pres, fin_choice]
    # Partial unfreeze path (stop early)
    clone, leaf = instance.path_clone([instance, is_pres_top, arg_node])
    assert id(clone) != id(instance)
    assert not clone.is_frozen
    assert clone != instance
    path = leaf.get_nodes_to_here()
    assert len(path) == 3
    new_arg_node: ObjectNode = leaf.cur_node
    assert new_arg_node.get_choice_node_for_arg(OPTIONAL_ARGUMENT_NEXT_ARG_NAME) is None


def test_objectnode_copy_with_2children():
    """Copypasta of the above test, just with an extra arg thrown in"""
    # Establish types
    tc = TypeContext()
    AInixType(tc, "footype")
    bartype = AInixType(tc, "bartype")
    arg1 = AInixArgument(tc, "arg1", "bartype")
    arg2 = AInixArgument(tc, "arg2", "bartype")
    foo_object = AInixObject(tc, "foo_object", "footype", [arg1, arg2])
    bar_object = AInixObject(tc, "bar_object", "bartype")
    # Make an ast
    fin_choice = ObjectNode(bar_object)
    is_pres = ObjectChoiceNode(bartype)
    is_pres.set_choice(fin_choice)
    arg_node = ObjectNode(arg1.is_present_object)
    arg_node.set_arg_value(OPTIONAL_ARGUMENT_NEXT_ARG_NAME, is_pres)
    is_pres_top = ObjectChoiceNode(arg1.present_choice_type)
    is_pres_top.set_choice(arg_node)
    instance = ObjectNode(foo_object)
    instance.set_arg_value("arg1", is_pres_top)

    fin_choice2 = ObjectNode(bar_object)
    is_pres2 = ObjectChoiceNode(bartype)
    is_pres2.set_choice(fin_choice2)
    arg_node2 = ObjectNode(arg2.is_present_object)
    arg_node2.set_arg_value(OPTIONAL_ARGUMENT_NEXT_ARG_NAME, is_pres2)
    is_pres_top2 = ObjectChoiceNode(arg2.present_choice_type)
    is_pres_top2.set_choice(arg_node2)
    instance.set_arg_value("arg2", is_pres_top2)
    # Do the tests:
    # Unfrozen
    clone, leaf_pointer = instance.path_clone()
    assert id(clone) != id(instance)
    assert clone == instance
    # Freeze part
    is_pres_top.freeze()
    clone, leaf_pointer = instance.path_clone()
    assert id(clone) != id(instance)
    assert not clone.is_frozen
    assert clone == instance
    assert id(clone.get_choice_node_for_arg("arg1")) == id(is_pres_top)
    assert id(clone.get_choice_node_for_arg("arg2")) != id(is_pres_top2)
    # Freeze all
    instance.freeze()
    clone, leaf_pointer = instance.path_clone()
    assert id(clone) == id(instance)
    assert clone == instance
    assert id(clone.get_choice_node_for_arg("arg1")) == id(is_pres_top)
    assert id(clone.get_choice_node_for_arg("arg2")) == id(is_pres_top2)
    # Full unfreeze path
    clone, leaf_pointer = instance.path_clone(
        [instance, is_pres_top, arg_node, is_pres, fin_choice])
    assert id(clone) != id(instance)
    assert not clone.is_frozen
    assert clone == instance
    assert leaf_pointer.get_nodes_to_here() == \
           [instance, is_pres_top, arg_node, is_pres, fin_choice]
    assert id(clone.get_choice_node_for_arg("arg2")) == id(is_pres_top2)
    assert clone.get_choice_node_for_arg("arg2").is_frozen
    # Partial unfreeze path (stop early)
    clone, leaf_pointer = instance.path_clone([instance, is_pres_top, arg_node])
    assert id(clone) != id(instance)
    assert not clone.is_frozen
    assert clone != instance
    assert len(leaf_pointer.get_nodes_to_here()) == 3
    new_arg_node: ObjectNode = leaf_pointer.cur_node
    assert new_arg_node.get_choice_node_for_arg(OPTIONAL_ARGUMENT_NEXT_ARG_NAME) is None
    assert new_arg_node.get_choice_node_for_arg(OPTIONAL_ARGUMENT_NEXT_ARG_NAME) is None
    assert clone.get_choice_node_for_arg("arg2") == is_pres_top2
    assert id(clone.get_choice_node_for_arg("arg2")) == id(is_pres_top2)


def test_dfs():
    tc = get_toy_strings_context()
    ast = ObjectChoiceNode(tc.get_type_by_name("ToySimpleStrs"))
    assert [n.cur_node for n in ast.depth_first_iter()] == [ast]


def test_dfs2():
    tc = get_toy_strings_context()
    two_strs = ObjectNode(tc.get_object_by_name("two_string"))
    assert [n.cur_node for n in two_strs.depth_first_iter()] == [two_strs]


def test_dfs3():
    tc = get_toy_strings_context()
    ast = ObjectChoiceNode(tc.get_type_by_name("ToySimpleStrs"))
    two_strs = ObjectNode(tc.get_object_by_name("two_string"))
    ast.set_choice(two_strs)
    assert [n.cur_node for n in ast.depth_first_iter()] == [ast, two_strs]


def test_dfs4():
    tc = get_toy_strings_context()
    ast = ObjectChoiceNode(tc.get_type_by_name("ToySimpleStrs"))

    two_strs = ObjectNode(tc.get_object_by_name("two_string"))
    ast.set_choice(two_strs)

    a1 = ObjectChoiceNode(tc.get_type_by_name("ToyMetasyntactic"))
    two_strs.set_arg_value("arg1", a1)

    assert [n.cur_node for n in ast.depth_first_iter()] == [ast, two_strs, a1]

    a1v = ObjectNode(tc.get_object_by_name("foo"))
    a1.set_choice(a1v)
    assert [n.cur_node for n in ast.depth_first_iter()] == [ast, two_strs, a1, a1v]


def test_dfs4half():
    tc = get_toy_strings_context()
    two_strs = ObjectNode(tc.get_object_by_name("two_string"))
    a1 = ObjectChoiceNode(tc.get_type_by_name("ToyMetasyntactic"))
    two_strs.set_arg_value("arg1", a1)
    a2 = ObjectChoiceNode(tc.get_type_by_name("ToyMetasyntactic"))
    two_strs.set_arg_value("arg2", a2)
    assert [n.cur_node for n in two_strs.depth_first_iter()] == [two_strs, a1, a2]


def test_dfs5():
    tc = get_toy_strings_context()
    ast = ObjectChoiceNode(tc.get_type_by_name("ToySimpleStrs"))
    two_strs = ObjectNode(tc.get_object_by_name("two_string"))
    ast.set_choice(two_strs)
    a1 = ObjectChoiceNode(tc.get_type_by_name("ToyMetasyntactic"))
    two_strs.set_arg_value("arg1", a1)
    a1v = ObjectNode(tc.get_object_by_name("foo"))
    a1.set_choice(a1v)

    a2 = ObjectChoiceNode(tc.get_type_by_name("ToyMetasyntactic"))
    two_strs.set_arg_value("arg2", a2)
    assert [n.cur_node for n in ast.depth_first_iter()] == [ast, two_strs, a1, a1v, a2]
    a2v = ObjectNode(tc.get_object_by_name("bar"))
    a2.set_choice(a2v)
    assert [n.cur_node for n in ast.depth_first_iter()] == [ast, two_strs, a1, a1v, a2, a2v]


@pytest.mark.parametrize("freeze_first", [False, True])
def test_pointer_change_here_root(freeze_first):
    node = ObjectChoiceNode(MagicMock())
    pointer = AstIterPointer(node, None, None)
    new_node = ObjectChoiceNode(MagicMock())
    if freeze_first:
        new_node.freeze()
    new_pointer = pointer.change_here(new_node)
    assert new_pointer.parent is None
    assert new_pointer.cur_node == new_node
    assert new_pointer.cur_node.is_frozen == freeze_first


@pytest.mark.parametrize("freeze_first", [False, True])
def test_pointer_change_here(freeze_first):
    """Test an arg change"""
    # Establish types
    tc = TypeContext()
    AInixType(tc, "footype")
    bartype = AInixType(tc, "bartype")
    arg1 = AInixArgument(tc, "arg1", "bartype", required=True)
    foo_object = AInixObject(tc, "foo_object", "footype", [arg1])
    bar_object = AInixObject(tc, "bar_object", "bartype")
    other_bar_obj = AInixObject(tc, "other_bar_ob", "bartype")
    # Make an ast
    arg_choice = ObjectChoiceNode(bartype)
    ob_chosen = ObjectNode(bar_object)
    arg_choice.set_choice(ob_chosen)
    instance = ObjectNode(foo_object)
    instance.set_arg_value("arg1", arg_choice)
    if freeze_first:
        instance.freeze()
    # Try change
    deepest = list(instance.depth_first_iter())[-1]
    assert deepest.cur_node == ob_chosen
    new_node = ObjectNode(other_bar_obj)
    new_point = deepest.change_here(new_node)
    assert new_point.cur_node == new_node
    assert new_point.cur_node.is_frozen == freeze_first


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
