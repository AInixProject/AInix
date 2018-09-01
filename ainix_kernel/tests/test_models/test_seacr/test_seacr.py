import pytest
from models.SeaCR.seacr import *
from typecontext import TypeContext
from ainix_common.parsing import loader

BUILTIN_TYPES_PATH = "../../../../builtin_types"


@pytest.fixture(scope="function")
def base_type_context():
    type_context = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/generic_parsers.ainix.yaml", type_context)
    return type_context


@pytest.fixture(scope="function")
def numbers_type_context(base_type_context):
    loader.load_path(f"{BUILTIN_TYPES_PATH}/numbers.ainix.yaml", base_type_context)
    base_type_context.fill_default_parsers()
    return base_type_context


def test_only_one_option(base_type_context):
    # Create some types
    foo_type = AInixType(base_type_context, "FooType")
    AInixObject(base_type_context, "FooObj", "FooType")
    base_type_context.fill_default_parsers()
    # Create an index
    index = ExamplesIndex(base_type_context, ExamplesIndex.get_default_ram_backend())
    index.add_many_to_many_default_weight(["example"], ["y"], index.DEFAULT_X_TYPE, "FooType")
    # Create a expected value
    parser = StringParser(foo_type)
    expected = parser.create_parse_tree("string")
    # Predict
    model = SeaCRModel(index)
    prediction = model.predict("example", "FooType")
    assert expected == prediction


def test_predict_digit(numbers_type_context):
    # Create an index
    index = ExamplesIndex(numbers_type_context, ExamplesIndex.get_default_ram_backend())
    x_y = [("one", "1"), ("two", "2"), ("three", "3")]
    for x, y in x_y:
        index.add_many_to_many_default_weight([x], [y], index.DEFAULT_X_TYPE, "BaseTen")
    # Create a expected value
    parser = StringParser(numbers_type_context.get_type_by_name("BaseTen"))
    expected = parser.create_parse_tree("2")
    # Predict
    model = SeaCRModel(index)
    prediction = model.predict("two", "BaseTen")
    assert expected == prediction


def test_digit_list_1(numbers_type_context):
    # Create an index
    index = ExamplesIndex(numbers_type_context, ExamplesIndex.get_default_ram_backend())
    type = "IntBase"
    x_y = [("one", "1"), ("two", "2"), ("three", "3")]
    for x, y in x_y:
        index.add_many_to_many_default_weight([x], [y], index.DEFAULT_X_TYPE, type)
    # Create a expected value
    parser = StringParser(numbers_type_context.get_type_by_name(type))
    expected = parser.create_parse_tree("2")
    # Predict
    model = SeaCRModel(index)
    prediction = model.predict("two", type)
    print(prediction.dump_str())
    print("expected")
    print(expected.dump_str())
    assert expected == prediction


def test_digit_list_2(numbers_type_context):
    # Create an index
    index = ExamplesIndex(numbers_type_context, ExamplesIndex.get_default_ram_backend())
    type = "IntBase"
    x_y = [("ten", "10"), ("twenty", "20"), ("thirty", "30")]
    for x, y in x_y:
        index.add_many_to_many_default_weight([x], [y], index.DEFAULT_X_TYPE, type)
    # Create a expected value
    parser = StringParser(numbers_type_context.get_type_by_name(type))
    expected = parser.create_parse_tree("20")
    # Predict
    model = SeaCRModel(index)
    prediction = model.predict("twenty", type)
    print(prediction.dump_str())
    print("expected")
    print(expected.dump_str())
    assert expected == prediction
