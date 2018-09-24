"""A bunch of tests of String to Tree models. Can be reused for multiple
models if new models are added. Each test will train the model with some data
and test specific aspects of the training"""
from typing import List

import pytest

from ainix_kernel.indexing.examplestore import Example, DataSplits, SPLIT_TYPE, ExamplesStore
from ainix_kernel.models.SeaCR import seacr
from ainix_kernel.indexing.exampleindex import ExamplesIndex
from ainix_common.parsing.typecontext import TypeContext, AInixType, AInixObject, AInixArgument
from ainix_common.parsing import loader
from ainix_kernel.models.model_types import StringTypeTranslateCF
from ainix_kernel.training.evaluate import EvaluateLogger
from ainix_kernel.training.train import TypeTranslateCFTrainer

# Here we define functions to generate each of the models we want to test
FULL_MODELS = ["SeaCR"]
ALL_MODELS = ["SeaCR-Rulebased"] + FULL_MODELS


def make_example_store(model_name, type_context):
    if model_name in ("SeaCR-Rulebased", "SeaCR"):
        return ExamplesIndex(type_context, ExamplesIndex.get_default_ram_backend())
    else:
        raise ValueError("Unrecognized model type ", type)


def make_model(model_name, example_store):
    if model_name == "SeaCR-Rulebased":
        return seacr.make_rulebased_seacr(example_store)
    elif model_name == "SeaCR":
        return seacr.make_default_seacr(example_store)
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
    AInixType(tc, "FooBarBazType", default_type_parser_name="max_munch_type_parser")
    AInixObject(tc, "foo", "FooBarBazType", type_data={"ParseRepresentation": "foo"})
    AInixObject(tc, "bar", "FooBarBazType", type_data={"ParseRepresentation": "bar"})
    AInixObject(tc, "baz", "FooBarBazType", type_data={"ParseRepresentation": "baz"})
    return tc


@pytest.fixture(scope="function")
def basic_string_tc(basic_classify_tc):
    tc = basic_classify_tc
    AInixType(tc, "FooStringType", default_object_parser_name="regex_group_object_parser")
    lhsArg = AInixArgument(tc, "lhs", "FooBarBazType", required=True,
                           arg_data={"RegexRepresentation": r"([a-z]+)"})
    rhsArg = AInixArgument(tc, "rhs", "FooStringType", required=False,
                           arg_data={"RegexRepresentation": r"[a-z]+ (.*)"})
    AInixObject(tc, "foo_string", "FooStringType", [lhsArg, rhsArg])
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


def do_train(model: StringTypeTranslateCF, example_store: ExamplesStore, epochs=10):
    trainer = TypeTranslateCFTrainer(model, example_store)
    trainer.train(epochs)


def assert_acc(model, example_store, splits, required_accuracy, expect_fail):
    """Checks whether a model achieves a certain accuracy on a specified split"""
    trainer = TypeTranslateCFTrainer(model, example_store)
    logger = EvaluateLogger()
    trainer.evaluate(logger, splits)
    if not expect_fail:
        assert logger.stats['ExactMatch'].true_frac >= required_accuracy
    else:
        assert not logger.stats['ExactMatch'].true_frac >= required_accuracy


def assert_train_acc(model, example_store, required_accuracy = 0.98, expect_fail=False):
    """Checks whether a model achieves a certain accuracy on the training split"""
    assert_acc(model, example_store, (DataSplits.TRAIN,), required_accuracy, expect_fail)


def assert_val_acc(model, example_store, required_accuracy = 0.98, expect_fail=False):
    """Checks whether a model achieves a certain accuracy on the validation split"""
    assert_acc(model, example_store, (DataSplits.VALIDATION,),
               required_accuracy, expect_fail)


@pytest.mark.parametrize("model_name", ALL_MODELS)  #, indirect=['model'])
def test_basic_classify(model_name, basic_classify_tc):
    basic_classify_tc.fill_default_parsers()
    example_store = make_example_store(model_name, basic_classify_tc)
    adder = ExampleAddHelper(example_store, ExamplesIndex.DEFAULT_X_TYPE,
                             "FooBarBazType", ALL_TRAIN_SPLIT)
    adder.add_examples(
        x_strings=["woof", "bow wo", "bark"],
        y_strings=["foo"]
    )
    adder.add_examples(
        x_strings=["meow", "prrr"],
        y_strings=["bar"],
    )
    adder.add_examples(
        x_strings=["moo", "im a cow"],
        y_strings=["baz"],
    )
    model = make_model(model_name, example_store)
    do_train(model, example_store)
    assert_train_acc(model, example_store)


@pytest.mark.parametrize("model_name", FULL_MODELS)
def test_non_bow(model_name, basic_classify_tc):
    """Tests to see if model can classify on tasks that require non-bag-of-words
    assumption"""
    basic_classify_tc.fill_default_parsers()
    example_store = make_example_store(model_name, basic_classify_tc)
    adder = ExampleAddHelper(example_store, ExamplesIndex.DEFAULT_X_TYPE,
                             "FooBarBazType",
                             ((0.7, DataSplits.TRAIN), (0.3, DataSplits.VALIDATION)))
    # start with bob
    adder.add_examples(
        x_strings=["bob bit the dog", "bob the bit dog", "bob bit dog the", "bob dog bit the"],
        y_strings=["foo"]
    )
    # start with dog
    adder.add_examples(
        x_strings=["dog bit the bob", "dog bit bob the", "dog bit bob the", "dog bit bit bob"],
        y_strings=["bar"],
    )
    # other
    adder.add_examples(
        x_strings=["the dog bit bob", "bit dog bob the", "bit bit bit bit", "bit dog"],
        y_strings=["baz"],
    )
    model = make_model(model_name, example_store)
    # Don't expect it to work before training.
    assert_val_acc(model, example_store, expect_fail=True)
    # Do training and expect it to work
    do_train(model, example_store, epochs=50)
    assert_train_acc(model, example_store)
    assert_val_acc(model, example_store)


@pytest.mark.parametrize("model_name", FULL_MODELS)
def test_string_gen(model_name, basic_string_tc):
    """Tests to see if model can learn to generate string of varying length"""
    basic_string_tc.fill_default_parsers()
    example_store = make_example_store(model_name, basic_string_tc)
    adder = ExampleAddHelper(example_store, ExamplesIndex.DEFAULT_X_TYPE,
                             "FooStringType",
                             ((0.8, DataSplits.TRAIN), (0.2, DataSplits.VALIDATION)))
    # if contains word "double", produce two of same type
    adder.add_examples(
        x_strings=["give me double", "double please", "I need a double",
                   "double double", "Gotta have double", "Where's my double",
                   "serve me up a double", "double trouble", "double the double",
                   "double down on it", "nothin but double", "double is where I need"],
        y_strings=["foo foo", "bar bar", "baz baz"]
    )
    # If contains the word single then just do a single one
    adder.add_examples(
        x_strings=["Give me a single", "single single", "gotta be single",
                   "all about that single", "big single", "top single",
                   "the real single", "yah know I want single", "single now",
                   "single life", "single sing single", "single pride",
                   "all single all the time"],
        y_strings=["foo", "bar", "baz"]
    )
    # if it contains confetti then should be foo
    adder.add_examples(
        x_strings=["confetti", "single confetti", "confetti party",
                   "one confetti party", "confetti poof", "look it's confetti",
                   "confetti is happy", "nonstop confetti"],
        y_strings=["foo"]
    )
    adder.add_examples(
        x_strings=["double confetti", "confetti party double",
                   "double the confetti party", "double confetti poof",
                   "look it's double confetti", "confetti is double happy",
                   "nonstop double confetti"],
        y_strings=["foo foo"]
    )
    # if contains "drunk" then should be bar
    adder.add_examples(
        x_strings=["so drunk", "wasted drunk", "wow that drunk sucked", "drunk on love",
                   "drunk on whiskey", "beer drunk", "what drunk last night"],
        y_strings=["bar"]
    )
    adder.add_examples(
        x_strings=["double drunk", "drunk double", "wow that double drunk sucked",
                   "I double drunk", "we double drunk", "is that double drunk"],
        y_strings=["bar bar"]
    )
    model = make_model(model_name, example_store)
    # Don't expect it to work before training.
    assert_val_acc(model, example_store, expect_fail=True)
    # Do training and expect it to work
    do_train(model, example_store)
    assert_train_acc(model, example_store)
    assert_val_acc(model, example_store)
