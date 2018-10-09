from abc import ABC, abstractmethod, abstractclassmethod
from ainix_common.parsing.parseast import ObjectChoiceNode, AstObjectChoiceSet
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import Example, ExamplesStore
from typing import Iterable, List


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


class StringTypeTranslateCF(Pretrainable):
    """Translates a string to another type without taking any prior context
    into account (what the user or the system has said previously)"""
    # TODO (DNGros): add interface for batched training and prediction

    @abstractmethod
    def predict(
        self,
        x_string: str,
        y_type_name: str,
        use_only_train_data: bool
    ) -> ObjectChoiceNode:
        pass

    @abstractmethod
    def train(self, x_string: str, y_ast: AstObjectChoiceSet,
              teacher_force_path: ObjectChoiceNode):
        pass

    @abstractclassmethod
    def make_examples_store(cls, type_context: TypeContext, is_training: bool) -> ExamplesStore:
        """Returns the an instance of the desired kind of example store"""
        raise NotImplemented

    # Some methods for communicating state during training. This is sort of
    # a bad interface. Should maybe abstract out into a seperate trainer class.

    def start_train_session(self):
        pass

    def end_train_session(self):
        pass

    def end_train_epoch(self):
        pass
