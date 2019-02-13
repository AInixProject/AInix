"""Centaurnix might eventually support many different kinds of models (currently
though the only supported model type converts from a string into an arbitrary
AST.) This module defines ABC's for models for the vary tasks and stuff that
might be useful."""
import math
from abc import ABC, abstractmethod, abstractclassmethod
from ainix_common.parsing.ast_components import ObjectChoiceNode, AstObjectChoiceSet
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import Example, ExamplesStore
from typing import Iterable, List, Tuple, Sequence, Optional
from ainix_common.parsing.model_specific import tokenizers
from pyrsistent import pvector
import attr

from ainix_kernel.model_util.lm_task_processor.lm_set_process import LMBatch


class ModelException(RuntimeError):
    """An exception for something that went wrong during a call in a model"""
    pass


class ModelValueError(ModelException):
    """An exception for when something goes wrong in the model due to the
    user providing malformed or inputs or inputs the model is not able to
    predict on"""
    pass


class ModelSafePredictError(ModelException):
    """An exception for when something goes wrong in the model but it is
    known failure mode. An example is reaching the max generation length
    while generating."""
    pass


class ModelCantPredictException(ModelException):
    """An exception for when the model received valid inputs, but is not able
    to make any prediction or judgment of confidence at all (likely due to
    the user submitting a query that is very different than any of the
    training data)."""
    pass


class Pretrainer(ABC):
    """A Pretrainer is used to update model parameters. Before running on new examples
    models can provide a Pretrainer implementation in their get_pretrainer method.
    The calling training method will ensure that each example is given to the
    pretrainer exactly once."""

    def __init__(self):
        self._is_open = True

    @abstractmethod
    def pretrain_example(self, example: Example, y_ast: ObjectChoiceNode):
        """Give an example to update parameters on. You should not give an example
        which this model has already been pretrained on."""
        if not self._is_open:
            raise ValueError("This pretrainer has already been closed.")

    def close(self):
        """After pretraining on all currently avaiable data, this method should
        be called. This allows the Pretrainer to  "finalize" whatever pretraining
        has been done. Different implementations may need to finalize in different
        ways or not at all. If no finalizing is needed, then this method can be
        left as the base implementation."""
        self._is_open = False


class Pretrainable(ABC):
    """An interface for something which can provide a pretrainer."""

    def get_pretrainer(self):
        return None


class RecusivePretrainer(Pretrainer):
    """A Pretrainer which doesn't actually do any pretraining itself, but takes
    a collection of pretrainers on init, and will pass all values to those."""

    def __init__(self, things_to_pretrain: Iterable[Pretrainable]):
        super().__init__()
        self._open_pretrainers: List[Pretrainer] = []
        for thing in things_to_pretrain:
            subpretrainer = thing.get_pretrainer()
            if subpretrainer is not None:
                self._open_pretrainers.append(subpretrainer)

    def pretrain_example(self, example: Example, y_ast: ObjectChoiceNode):
        for pretrainer in self._open_pretrainers:
            pretrainer.pretrain_example(example, y_ast)

    def close(self):
        for pretrainer in self._open_pretrainers:
            pretrainer.close()


class Model(ABC):
    """The shared interface parts for all models which we support"""
    # Some methods for communicating state during training. This is sort of
    # a bad interface. Should maybe abstract out into a seperate trainer class.

    def start_train_session(self):
        """Create optimizers and such to be able to train"""
        pass

    def end_train_session(self):
        """Free any optimizers and stuff from training"""
        pass

    def end_train_epoch(self):
        pass

    def set_in_train_mode(self):
        """This sets the parameters to act like training. This is different from
        a session as a model might go in and out of training mode during one session
        (for example if when evaling at the end of an epoch)"""
        pass

    def set_in_eval_mode(self):
        """Sets the parameters to act like eval (for example disabling dropout.)
        This may happen during a training session."""
        pass

    # Methods for handling serialization
    def get_save_state_dict(self):
        """Returns a dict which can be seriallized and later used to restore this
        model"""
        raise NotImplemented

    @classmethod
    def create_from_save_state_dict(
            cls,
            state_dict: dict,
            new_type_context: TypeContext,
            new_example_store: ExamplesStore
    ):
        raise NotImplemented


class StringTypeTranslateCF(Model, Pretrainable):
    """Translates a string to another type without taking any prior context
    into account (what the user or the system has said previously)"""

    @abstractmethod
    def predict(
        self,
        x_string: str,
        y_type_name: str,
        use_only_train_data: bool
    ) -> Tuple[ObjectChoiceNode, 'TypeTranslatePredictMetadata']:
        raise NotImplemented

    @abstractmethod
    def train(
        self,
        x_string: str,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode,
        example_id: int
    ):
        raise NotImplemented

    def train_batch(
        self,
        batch: List[Tuple[str, AstObjectChoiceSet, ObjectChoiceNode, int]]
    ):
        """Used for batch training. Will by default call train on each element
        in batch, but expected to be overridden.

        Args:
            batch: List of tuples which have 3 values:
                x: the string x value
                y: the ast set we are trying to generate to
                teacher_force_paths: the path to follow while decide
                example_id: the example id this came from
        """
        for x, y, path, example_id in zip(batch):
            self.train(x, y, path, example_id)

    @classmethod
    @abstractmethod
    def make_examples_store(cls, type_context: TypeContext, is_training: bool) -> ExamplesStore:
        """Returns the an instance of the desired kind of example store"""
        raise NotImplemented

    def set_shared_memory(self):
        """Used for when doing multiprocess training to set to shared memory."""
        raise NotImplemented

    @abstractmethod
    def get_string_tokenizer(self) -> tokenizers.StringTokenizer:
        pass


@attr.s(auto_attribs=True, frozen=True)
class TypeTranslatePredictMetadata:
    """
    Args:
        log_confidences: A sequence which for ObjectChoiceNode choice in the prediction
            Gives  A value less < 0. When raised to e power, will be in the
            interval [0, 1]. This should ideally roughly correspond to a
            probability, but in reality this currently will probably be poorly
            calibrated, or the loss might be formulated that a probability
            interpretation does not hold. In the future we need to work on
            improving the interpretability guarrantees of this number.
    """
    log_confidences: Tuple[float, ...]
    example_retrieve_explanations: Tuple['ExampleRetrieveExplanation', ...]

    @property
    def total_confidence(self) -> float:
        return math.exp(sum(self.log_confidences))

    @classmethod
    def create_leaf_value(
        cls,
        log_confidence: float,
        example_retrieve_explantion: 'ExampleRetrieveExplanation' = None
    ):
        return cls(
            log_confidences=(log_confidence,),
            example_retrieve_explanations=(example_retrieve_explantion,)
                if example_retrieve_explantion is not None else tuple()
        )

    @classmethod
    def create_empty(cls):
        return cls(
            log_confidences=tuple(),
            example_retrieve_explanations=tuple()
        )

    def concat(self, later_vals: 'TypeTranslatePredictMetadata') -> 'TypeTranslatePredictMetadata':
        """Immutably returns a instance which combines this data with other data at end"""
        return TypeTranslatePredictMetadata(
            log_confidences=self.log_confidences + later_vals.log_confidences,
            example_retrieve_explanations=self.example_retrieve_explanations +
                                          later_vals.example_retrieve_explanations
        )


@attr.s(auto_attribs=True, frozen=True)
class ExampleRetrieveExplanation:
    """Contains several examples from the example index which support our choice
    at this step."""
    reference_example_ids: Tuple[int, ...]
    reference_confidence: Tuple[float, ...]
    reference_example_dfs_ind: Tuple[int, ...]


class BertlikeLangModel(Model):
    """A model that which model which performance a language modeling task
    similar to that to BERT (https://arxiv.org/abs/1810.04805).

    This includes whether predicting two sentences are sequentially drawn from
    the same document as well as predicting several masked out tokens.
    """
    def eval_run(
        self,
        batch: LMBatch
    ):
        raise NotImplemented()

    def train_batch(
        self,
        batch: LMBatch
    ):
        raise NotImplemented()
