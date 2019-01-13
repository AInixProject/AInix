"""A bunch of tests of String2Tree models. Can be reused for multiple
models if new models are added. Each test will train the model with some data
and test specific aspects of the training.

These tests can be somewhat flaky. Look into making this better perhaps with set
seeds."""
import os
import sys
import torch
from typing import List

import pytest

from ainix_common.parsing.grammar_lang import create_object_parser_from_grammar
from ainix_kernel.indexing.examplestore import Example, DataSplits, SPLIT_TYPE, ExamplesStore
from ainix_kernel.models.EncoderDecoder import encdecmodel
from ainix_kernel.models.SeaCR import seacr
from ainix_kernel.indexing.exampleindex import ExamplesIndex
from ainix_common.parsing.typecontext import TypeContext, AInixType, AInixObject, AInixArgument
from ainix_common.parsing import loader
from ainix_kernel.models.model_types import StringTypeTranslateCF
from ainix_kernel.training.evaluate import EvaluateLogger
import tempfile

# Here we define functions to generate each of the models we want to test
# Full models are models which should pass every test
#FULL_MODELS = ["SeaCR"]
from ainix_kernel.training.trainer import TypeTranslateCFTrainer

FULL_MODELS = ["EncDec", "EncDecRetrieval"]
# All Models are just all available models. Some might not be expect to pass
# every test
#ALL_MODELS = ["SeaCR-Rulebased", "SeaCR-NoSearch", "SeaCR-OracleCompare"] + FULL_MODELS
ALL_MODELS = FULL_MODELS


def make_example_store(model_name, type_context):
    if model_name in ("SeaCR-Rulebased", "SeaCR", "SeaCR-OracleCompare", "SeaCR-NoSearch"):
        return ExamplesIndex(type_context, ExamplesIndex.get_default_ram_backend())
    if model_name in ("EncDec", "EncDecRetrieval"):
        # TODO (DNGros): avoid the whoosh stuff
        return ExamplesIndex(type_context, ExamplesIndex.get_default_ram_backend())
    else:
        raise ValueError("Unrecognized model type ", type)


def make_model(model_name, example_store):
    if model_name == "SeaCR-Rulebased":
        return seacr.make_rulebased_seacr(example_store)
    elif model_name == "SeaCR":
        return seacr.make_default_seacr(example_store)
    elif model_name == "SeaCR-NoSearch":
        return seacr.make_default_seacr_no_search(example_store)
    elif model_name == "SeaCR-OracleCompare":
        return seacr.make_default_seacr_with_oracle_comparer(example_store)
    elif model_name == "EncDec":
        return encdecmodel.get_default_encdec_model(example_store)
    elif model_name == "EncDecRetrieval":
        return encdecmodel.get_default_encdec_model(
            example_store, replacer=None, use_retrieval_decoder=True)
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
    for word in ("foo", "baz", "bar", "pop"):
        AInixObject(tc, word, "FooBarBazType",
                    preferred_object_parser_name=create_object_parser_from_grammar(
                        tc, f"par{word}", f'"{word}"').name)
    return tc


@pytest.fixture(scope="function")
def basic_string_tc(basic_classify_tc):
    tc = basic_classify_tc
    AInixType(tc, "FooStringType")
    lhsArg = AInixArgument(tc, "lhs", "FooBarBazType", required=True, parent_object_name="wer")
    rhsArg = AInixArgument(tc, "rhs", "FooStringType", required=False, parent_object_name="sdf")
    AInixObject(tc, "foo_string", "FooStringType", [lhsArg, rhsArg],
                preferred_object_parser_name=create_object_parser_from_grammar(
                    tc, "itasdf", 'lhs (" " rhs)?'
                ).name)
    return tc


ALL_TRAIN_SPLIT: SPLIT_TYPE = ((1, DataSplits.TRAIN),)
ALL_VAL_SPLIT: SPLIT_TYPE = ((1, DataSplits.VALIDATION),)


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


def do_train(
    model: StringTypeTranslateCF,
    example_store: ExamplesStore,
    epochs: int = 10,
    batch_size: int = 1
):
    trainer = TypeTranslateCFTrainer(model, example_store, batch_size)
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


def check_survives_serialization(
        model: StringTypeTranslateCF, example_store, new_type_context: TypeContext,
        acc=0.98, only_test_train=False):
    save_state = model.get_save_state_dict()
    _, f = tempfile.mkstemp()
    try:
        torch.save(save_state, f)
        new_state = torch.load(f)
        new_model = model.create_from_save_state_dict(new_state, new_type_context, example_store)
        assert_train_acc(new_model, example_store, acc)
        if not only_test_train:
            assert_val_acc(new_model, example_store, acc)
    finally:
        os.remove(f)


@pytest.mark.parametrize("model_name", ALL_MODELS)  #, indirect=['model'])
def test_basic_classify(model_name, basic_classify_tc):
    basic_classify_tc.finalize_data()
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
    do_train(model, example_store, epochs=100)
    assert_train_acc(model, example_store)


@pytest.mark.parametrize("model_name", ALL_MODELS)  #, indirect=['model'])
def test_basic_classify_serialize(model_name, basic_classify_tc):
    # TODO don't require retraining each time.
    basic_classify_tc.finalize_data()
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
    do_train(model, example_store, epochs=100)
    check_survives_serialization(model, example_store, basic_classify_tc, only_test_train=True)


@pytest.mark.parametrize("model_name", FULL_MODELS)
def test_classify_seq(model_name, basic_string_tc):
    """Tests to see if model can learn to generate string of varying length"""
    basic_string_tc.finalize_data()
    example_store = make_example_store(model_name, basic_string_tc)
    adder = ExampleAddHelper(example_store, ExamplesIndex.DEFAULT_X_TYPE,
                             "FooStringType", ALL_TRAIN_SPLIT)
    adder.add_examples(
        x_strings=["woof", "bow wo", "bark"],
        y_strings=["foo foo"]
    )
    adder.add_examples(
        x_strings=["meow", "prrr"],
        y_strings=["foo bar"],
    )
    adder.add_examples(
        x_strings=["moo", "im a cow"],
        y_strings=["bar foo"],
    )
    adder.add_examples(
        x_strings=["bloop"],
        y_strings=["foo"],
    )
    model = make_model(model_name, example_store)
    do_train(model, example_store, epochs=200)
    assert_train_acc(model, example_store)
    check_survives_serialization(model, example_store, basic_string_tc, only_test_train=True)


#@pytest.mark.parametrize("model_name", ["SeaCR-OracleCompare", "SeaCR"] + FULL_MODELS)
@pytest.mark.parametrize("model_name", FULL_MODELS)
def test_non_bow(model_name, basic_classify_tc):
    """Tests to see if model can classify on tasks that require expressive power
    beyond a bag-of-words assumption"""
    basic_classify_tc.finalize_data()
    example_store = make_example_store(model_name, basic_classify_tc)
    adder = ExampleAddHelper(example_store, ExamplesIndex.DEFAULT_X_TYPE,
                             "FooBarBazType", ALL_TRAIN_SPLIT)
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
        x_strings=["the dog bit bob", "bit dog bob the", "bit the dog bit", "bit dog the bob"],
        y_strings=["baz"],
    )
    model = make_model(model_name, example_store)
    print("train_len", len(list(
        example_store.get_all_examples(filter_splits=(DataSplits.TRAIN,)))))
    print("val_len", len(list(
        example_store.get_all_examples(filter_splits=(DataSplits.VALIDATION,)))))
    print("val examples", [x.xquery for x in
        example_store.get_all_examples(filter_splits=(DataSplits.VALIDATION,))])
    if "Oracle" not in model_name:
        # Don't expect it to work before training.
        assert_train_acc(model, example_store, expect_fail=True)
    # Do training and expect it to work
    do_train(model, example_store, epochs=100)
    assert_train_acc(model, example_store, required_accuracy=0.95)


@pytest.mark.parametrize("model_name", FULL_MODELS)
def test_string_gen(model_name, basic_string_tc):
    """Tests to see if model can learn to generate string of varying length"""
    basic_string_tc.finalize_data()
    example_store = make_example_store(model_name, basic_string_tc)
    adder = ExampleAddHelper(example_store, ExamplesIndex.DEFAULT_X_TYPE,
                             "FooStringType",
                             #ALL_TRAIN_SPLIT)
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
                   "drunk on whiskey", "beer drunk", "what drunk last night", "drunk"],
        y_strings=["bar"]
    )
    adder.add_examples(
        x_strings=["double drunk", "drunk double", "wow that double drunk sucked",
                   "I double drunk", "we double drunk", "is that double drunk"],
        y_strings=["bar bar"]
    )
    import torch
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    model = make_model(model_name, example_store)
    if "Oracle" not in model_name:
        # Don't expect it to work before training.
        #assert_val_acc(model, example_store, expect_fail=True)
        pass
    # Do training and expect it to work
    do_train(model, example_store, epochs=25, batch_size=1)
    assert_train_acc(model, example_store, required_accuracy=0.85)
    assert_val_acc(model, example_store, required_accuracy=0.7)


@pytest.mark.parametrize("model_name", FULL_MODELS)
def test_copy(model_name, basic_string_tc):
    basic_string_tc.finalize_data()
    example_store = make_example_store(model_name, basic_string_tc)
    adder = ExampleAddHelper(example_store, ExamplesIndex.DEFAULT_X_TYPE,
                             "FooStringType", ALL_TRAIN_SPLIT)
    for x, y in [
        ('hello there "foo bar baz foo"', "foo bar baz foo"),
        ('hello there "bar bar foo foo"', "bar bar foo foo"),
        ('hello there "baz bar foo baz"', "baz bar foo baz"),
        ('hey there "baz baz bar baz"', "baz baz bar baz"),
        ('hey there "foo bar baz baz"', "foo bar baz baz"),
        ('hey there "foo foo foo baz"', "foo foo foo baz"),
        ('hey there "bar bar baz foo"', "bar bar baz foo"),
        ('hello there "foo bar bar baz"', "foo bar bar baz"),
        ('hey there "bar bar bar baz"', "bar bar bar baz"),
    ]:
        adder.add_examples([x], [y], insert_splits=ALL_TRAIN_SPLIT)
    for x, y in [
        ('hello there "bar foo baz foo"', "bar foo baz foo"),
        ('hello there "foo foo foo baz"', "foo foo foo baz"),
        ('hey there "baz bar bar foo"', "baz bar bar foo"),
        ('hey there "baz foo foo baz"', "baz foo foo baz"),
    ]:
        adder.add_examples([x], [y], insert_splits=ALL_VAL_SPLIT)
    import torch
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    model = make_model(model_name, example_store)
    if "Oracle" not in model_name:
        # Don't expect it to work before training.
        assert_val_acc(model, example_store, expect_fail=True)
    # Do training and expect it to work
    do_train(model, example_store, epochs=30, batch_size=1)
    assert_train_acc(model, example_store, required_accuracy=0.85)
    assert_val_acc(model, example_store, required_accuracy=0.8)
    print("PASS NON-SERIALIZED")
    check_survives_serialization(model, example_store, basic_string_tc, acc=0.8)

