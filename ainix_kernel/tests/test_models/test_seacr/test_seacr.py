import pytest

import ainix_kernel.models.SeaCR.type_predictor
import models
from ainix_kernel.models.SeaCR.seacr import *
from ainix_kernel.models.SeaCR import seacr
from ainix_common.parsing.typecontext import TypeContext, AInixObject
from ainix_common.parsing import loader
import indexing.exampleloader
from ainix_common.parsing.parseast import StringParser
from ainix_kernel.models.model_types import ModelCantPredictException

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
    parser = StringParser(base_type_context)
    expected = parser.create_parse_tree("string", foo_type.name)
    # Predict
    model = make_rulebased_seacr(index)
    prediction = model.predict("example", "FooType", False)
    assert expected == prediction


def test_predict_digit(numbers_type_context):
    # Create an index
    index = ExamplesIndex(numbers_type_context, ExamplesIndex.get_default_ram_backend())
    x_y = [("one", "1"), ("two", "2"), ("three", "3")]
    for x, y in x_y:
        index.add_many_to_many_default_weight([x], [y], index.DEFAULT_X_TYPE, "BaseTen")
    # Create a expected value
    parser = StringParser(numbers_type_context)
    expected = parser.create_parse_tree("2", "BaseTen")
    # Predict
    model = make_rulebased_seacr(index)
    prediction = model.predict("two", "BaseTen", False)
    assert expected == prediction


def test_digit_list_1(numbers_type_context):
    # Create an index
    index = ExamplesIndex(numbers_type_context, ExamplesIndex.get_default_ram_backend())
    type = "IntBase"
    x_y = [("one", "1"), ("two", "2"), ("three", "3")]
    for x, y in x_y:
        index.add_many_to_many_default_weight([x], [y], index.DEFAULT_X_TYPE, type)
    # Create a expected value
    parser = StringParser(numbers_type_context)
    expected = parser.create_parse_tree("2", type)
    # Predict
    model = make_rulebased_seacr(index)
    prediction = model.predict("two", type, False)
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
    parser = StringParser(numbers_type_context)
    expected = parser.create_parse_tree("20", type)
    # Predict
    model = make_rulebased_seacr(index)
    prediction = model.predict("twenty", type, False)
    print(expected.dump_str())
    print(prediction.dump_str())
    assert expected == prediction


def test_full_number(numbers_type_context):
    # Create an index
    index = ExamplesIndex(numbers_type_context, ExamplesIndex.get_default_ram_backend())
    type = "Number"
    x_y = [("one", "1"), ("two", "2"), ("three", "3"), ("ten", "10"), ("negative one", "-1")]
    for x, y in x_y:
        index.add_many_to_many_default_weight([x], [y], index.DEFAULT_X_TYPE, type)
    # Create a expected value
    parser = StringParser(numbers_type_context)
    expected = parser.create_parse_tree("-1", type)
    # Predict
    model = make_rulebased_seacr(index)
    prediction = model.predict("negative one", type, False)
    print("Expected")
    print(expected.dump_str())
    print("predicted")
    print(prediction.dump_str())
    assert expected.dump_str() == prediction.dump_str()
    assert expected == prediction
    #
    expected = parser.create_parse_tree("2", type)
    prediction = model.predict("two", type, False)
    assert expected == prediction
    #
    with pytest.raises(ModelCantPredictException):
        model.predict("sdsdfas sdfasdf asdf", type, False)


def test_full_number_2(numbers_type_context):
    # Create an index
    index = ExamplesIndex(numbers_type_context, ExamplesIndex.get_default_ram_backend())
    indexing.exampleloader.load_path(
        f"{BUILTIN_TYPES_PATH}/numbers_examples.ainix.yaml", index)
    # Create a expected value
    parser = StringParser(numbers_type_context)
    expected = parser.create_parse_tree("9", "Number")
    # Predict
    model = make_rulebased_seacr(index)
    prediction = model.predict("nineth", "Number", False)
    assert expected == prediction


def test_type_pred_gt_result(numbers_type_context):
    parser = StringParser(numbers_type_context)
    ast = parser.create_parse_tree("9", "BaseTen")
    type = numbers_type_context.get_type_by_name("BaseTen")
    valid_set = AstObjectChoiceSet(type, None)
    valid_set.add(ast, True, 1, 1)
    choose = ObjectChoiceNode(type)
    gt_res = models.SeaCR.type_predictor._create_gt_compare_result(ast, choose, valid_set)
    assert gt_res.prob_valid_in_example == 1
    assert gt_res.impl_scores == ((1, "nine"),)
    ast = parser.create_parse_tree("6", "BaseTen")
    gt_res = models.SeaCR.type_predictor._create_gt_compare_result(ast, choose, valid_set)
    assert gt_res.prob_valid_in_example == 0
    assert gt_res.impl_scores is None
    # TODO (DNGros): add more tests of this
