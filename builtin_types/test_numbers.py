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


def test_number_end_to_end(type_context):
    numbers_type = type_context.get_type_by_name("Number")
    obj_by_name = type_context.get_object_by_name
    parser = StringParser(numbers_type)
    result = parser.create_parse_tree("4")
    weight = 1
    # build expected tree
    expected = ObjectChoiceNode(numbers_type, None)
    dec_number_node = expected.add_valid_choice(
        obj_by_name("decimal_number"), weight)
    #sign_choice = dec_number_node.set_arg_present(
    #    dec_number_node.implementation.children[0])
    #sign_object = sign_choice.add_valid_choice(
    #    type_context.get_object_by_name("positive"), weight)
    before_decimal_choice = dec_number_node.set_arg_present(
        dec_number_node.implementation.children[1])
    num_list_obj = before_decimal_choice.add_valid_choice(
        obj_by_name("base_ten_list"), weight)
    digit_choice = num_list_obj.set_arg_present("CurrentDigit", weight)
    digit_choice.add_valid_choice(obj_by_name("four"), weight)
    # compare
    assert result.dump_str() == expected.dump_str()
    assert result == expected
