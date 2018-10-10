import pytest
from indexing.exampleindex import *
import ainix_common.parsing.ast_components
from indexing.examplestore import Example
from ainix_common.parsing import loader
from typecontext import AInixType


def test_nearest_example():
    type_context = TypeContext()
    type_name = "DeepThoughts"
    other_type_name = "PythonThoughts"
    top_type = AInixType(type_context, type_name)
    other_type = AInixType(type_context, other_type_name)
    parsed_rep = ainix_common.parsing.ast_components.indexable_repr_classify_type(top_type.name)
    print(parsed_rep)
    other_parsed_rep = ainix_common.parsing.ast_components.indexable_repr_classify_type(other_type.name)
    print(other_parsed_rep)

    index = ExamplesIndex(type_context, backend=ExamplesIndex.get_default_ram_backend())
    index.add_example(Example("what is the meaning of life", "42",
                              index.DEFAULT_X_TYPE, type_name, 1,
                              "sdff", DataSplits.TRAIN.value, parsed_rep))
    index.add_example(Example("what is the meaning of the universe", "forty two",
                              index.DEFAULT_X_TYPE, type_name, 1,
                              "bdffg", DataSplits.TRAIN.value, parsed_rep))
    index.add_example(Example("what... is the airspeed of an unladen swallow?",
                              "African or European?", index.DEFAULT_X_TYPE,
                              other_type_name, 1,
                              "sfasdf", DataSplits.TRAIN.value, other_parsed_rep))
    index.add_example(Example("what is the meaning of everything else", "four two",
                              index.DEFAULT_X_TYPE, type_name, 1,
                              "sdfasdf", DataSplits.TRAIN.value, parsed_rep))
    index.add_example(Example("what... is your quest?",
                              "To seek the Holy Grail", index.DEFAULT_X_TYPE,
                              other_type_name, 1, "sdfanb",
                              DataSplits.TRAIN.value, other_parsed_rep))
    #result = list(index.get_nearest_examples("what... is your quest?", other_type.name))
    #print(result)
    #assert len(result) == 2
    #assert result[0].xquery == "what... is your quest?"

    result = list(index.get_nearest_examples(
        "what... is your quest?", other_type.name, max_results=1))
    assert len(result) == 1
    assert result[0].xquery == "what... is your quest?"
    result = list(index.get_nearest_examples(
        "everything else", top_type.name, max_results=1))
    assert len(result) == 1
    assert result[0].xquery == "what is the meaning of everything else"


def test_get_all_docs():
    type_context = TypeContext()
    type_name = "DeepThoughts"
    other_type_name = "PythonThoughts"
    top_type = AInixType(type_context, type_name)
    other_type = AInixType(type_context, other_type_name)
    parsed_rep = ainix_common.parsing.ast_components.indexable_repr_classify_type(top_type.name)
    other_parsed_rep = ainix_common.parsing.ast_components.indexable_repr_classify_type(other_type.name)

    index = ExamplesIndex(type_context, backend=ExamplesIndex.get_default_ram_backend())
    index.add_example(Example("what is the meaning of life", "42",
                              index.DEFAULT_X_TYPE, type_name, 1, "sdf",
                              DataSplits.TRAIN.value, parsed_rep))
    index.add_example(Example("what is the meaning of the universe", "forty two",
                              index.DEFAULT_X_TYPE, type_name, 1, "sdfnb",
                              DataSplits.TEST.value, parsed_rep))
    index.add_example(Example("what... is the airspeed of an unladen swallow?",
                              "African or European?", index.DEFAULT_X_TYPE,
                              other_type_name, 1, "basdf",
                              DataSplits.VALIDATION.value, other_parsed_rep))
    index.add_example(Example("what is the meaning of everything else", "four two",
                              index.DEFAULT_X_TYPE, type_name, 1, "sdbf",
                              DataSplits.TRAIN.value, parsed_rep))
    index.add_example(Example("what... is your quest?",
                              "To seek the Holy Grail", index.DEFAULT_X_TYPE,
                              other_type_name, 1, "nwand",
                              DataSplits.VALIDATION.value, other_parsed_rep))

    assert len(list(index.get_all_examples())) == 5
    assert len(list(index.get_all_examples((DataSplits.VALIDATION,)))) == 2
    assert len(list(index.get_all_examples((DataSplits.VALIDATION, DataSplits.TRAIN)))) == 4


BUILTIN_TYPES_PATH = "../../../builtin_types"


@pytest.fixture(scope="function")
def base_type_context():
    type_context = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/generic_parsers.ainix.yaml", type_context)
    return type_context


@pytest.fixture(scope="function")
def numbers_context(base_type_context):
    loader.load_path(f"{BUILTIN_TYPES_PATH}/numbers.ainix.yaml", base_type_context)
    base_type_context.fill_default_parsers()
    return base_type_context


def test_y_set(numbers_context):
    index = ExamplesIndex(numbers_context, backend=ExamplesIndex.get_default_ram_backend())
    index.add_many_to_many_default_weight(["ten"], ["10", "1e1"],
                                          index.DEFAULT_X_TYPE, "Number")
    index.add_many_to_many_default_weight(["five"], ["5"],
                                          index.DEFAULT_X_TYPE, "Number")
    index.add_many_to_many_default_weight(["one hundred", "hundred"], ["100", "10e1", "1e2"],
                                          index.DEFAULT_X_TYPE, "Number")
    nearest = list(index.get_nearest_examples("ten"))
    assert len(nearest) == 2  # 2 since it does a many to many map. This might change
    example = nearest[0]
    assert example.xquery == "ten"
    y_set = list(index.get_examples_from_y_set(example.y_set_id))
    assert len(y_set) == 2
    y_val_set = {e.ytext for e in y_set}
    assert "10" in y_val_set
    assert "1e1" in y_val_set
