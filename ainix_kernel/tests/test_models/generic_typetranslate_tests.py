"""A bunch of tests of String to Tree models. Can be reused for multiple
models if new models are added. Each test will train the model with some data
and test specific aspects of the training"""
from typing import List

import pytest

from indexing.examplestore import Example, DataSplits, SPLIT_TYPE, ExamplesStore
from models.SeaCR.seacr import SeaCRModel
from indexing.exampleindex import ExamplesIndex
from ainix_common.parsing.typecontext import TypeContext, AInixType, AInixObject
from ainix_common.parsing import loader
from models.model_types import StringTypeTranslateCF
from training.evaluate import EvaluateLogger
from training.train import TypeTranslateCFTrainer

# Here we define functions to generate each of the models we want to test
AVAIL_MODELS = ["SeaCR"]


def make_example_store(model_name, type_context):
    if model_name == "SeaCR":
        return ExamplesIndex(type_context, ExamplesIndex.get_default_ram_backend())
    else:
        raise ValueError("Unrecognized model type ", type)

def make_model(model_name, example_store):
    if model_name == "SeaCR":
        return SeaCRModel(example_store)
    else:
        raise ValueError("Unrecognized model type ", type)

# A bunch of type contexts to test on
BUILTIN_TYPES_PATH = "../../../builtin_types"


@pytest.fixture(scope="function")
def base_tc():
    tc = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/generic_parsers.ainix.yaml", tc)
    return tc


@pytest.fixture(scope="function")
def basic_classify_tc(base_tc):
    tc = base_tc
    AInixType(tc, "TopType", default_type_parser_name="max_munch_type_parser")
    AInixObject(tc, "dog", "TopType", type_data={"ParseRepresentation": "dog"})
    AInixObject(tc, "cat", "TopType", type_data={"ParseRepresentation": "cat"})
    AInixObject(tc, "cow", "TopType", type_data={"ParseRepresentation": "cow"})
    tc.fill_default_parsers()
    return tc


ALL_TRAIN_SPLIT: SPLIT_TYPE = ((1, DataSplits.TRAIN),)

class ExampleAddHelper:
    """Helps when adding a bunch of examples"""
    def __init__(
        self,
        example_store: ExamplesStore,
        default_x_type: str,
        default_y_type: str,
        default_insert_splits: SPLIT_TYPE
    ):
        self.example_store = example_store
        self.default_x_type = default_x_type
        self.default_y_type = default_y_type
        self.default_insert_splits = default_insert_splits

    def add_examples(
        self,
        x_strings: List[str],
        y_strings: List[str],
        x_type: str = None,
        y_type: str = None,
        insert_splits: SPLIT_TYPE = None
    ):
        self.example_store.add_many_to_many_default_weight(
            x_values=x_strings,
            y_values=y_strings,
            x_type=x_type or self.default_x_type,
            y_type=y_type or self.default_y_type,
            splits=insert_splits or self.default_insert_splits
        )


def do_train(model: StringTypeTranslateCF, example_store: ExamplesStore):
    trainer = TypeTranslateCFTrainer(model, example_store)
    trainer.train(10)


def assert_train_acc(model, example_store, required_accuracy = 0.98):
    """Checks whether a model achieves a certain accuracy on the training split"""
    trainer = TypeTranslateCFTrainer(model, example_store)
    logger = EvaluateLogger()
    trainer.evaluate(logger, (DataSplits.TRAIN,))
    assert logger.stats['ExactMatch'].true_frac >= required_accuracy


@pytest.mark.parametrize("model_name", AVAIL_MODELS)  #, indirect=['model'])
def test_basic_classify(model_name, basic_classify_tc):
    example_store = make_example_store(model_name, basic_classify_tc)
    adder = ExampleAddHelper(example_store, ExamplesIndex.DEFAULT_X_TYPE,
                             "TopType", ALL_TRAIN_SPLIT)
    adder.add_examples(
        x_strings=["woof", "bow wo", "bark"],
        y_strings=["dog"]
    )
    adder.add_examples(
        x_strings=["meow", "prrr"],
        y_strings=["cat"],
    )
    adder.add_examples(
        x_strings=["moo", "im a cow"],
        y_strings=["cow"],
    )
    model = make_model(model_name, example_store)
    do_train(model, example_store)
    assert_train_acc(model, example_store)


