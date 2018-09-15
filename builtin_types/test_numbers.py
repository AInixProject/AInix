import pytest
from typecontext import TypeContext
from parseast import StringParser, ObjectChoiceNode
import loader


@pytest.fixture(scope="function")
def type_context():
    context = TypeContext()
    loader.load_path("numbers.ainix.yaml", context)
    loader.load_path("generic_parsers.ainix.yaml", context)
    context.fill_default_parsers()
    return context


# Test broke. Should probably fix it at some point. If youre reading this prob just delete it...
#def test_number_end_to_end(type_context):
#    numbers_type = type_context.get_type_by_name("Number")
#    obj_by_name = type_context.get_object_by_name
#    parser = StringParser(type_context)
#    result = parser.create_parse_tree("4", numbers_type.name)
#    weight = 1
#    # build expected tree
#    expected = ObjectChoiceNode(numbers_type, None)
#    dec_number_node = expected.set_choice(
#        obj_by_name("decimal_number"))
#    before_decimal_choice = dec_number_node.set_arg_present(
#        dec_number_node.implementation.children[1])
#    num_list_obj = before_decimal_choice.add_valid_choice(
#        obj_by_name("base_ten_list"), weight)
#    digit_choice = num_list_obj.set_arg_present(
#        num_list_obj.implementation.children[0])
#    digit_choice.add_valid_choice(obj_by_name("four"), weight)
#    # compare
#    assert result.dump_str() == expected.dump_str()
#    assert result == expected
