from abc import ABC, abstractmethod
from parseast import ObjectChoiceNode, AstObjectChoiceSet


class ModelException(RuntimeError):
    """An exception for something that went wrong during model"""
    pass


class ModelValueError(ModelException):
    """An exception for when something goes wrong in the model due to the
    user providing malformed or inputs or inputs the model is not able to
    predict on"""
    pass


class ModelCantPredictException(ModelException):
    """An exception for when the model received valid inputs, but is not able
    to make any prediction or judgment of confidence at all (likely due to
    the user submitting a query that is very different than any of the
    training data)"""
    pass


class StringTypeTranslateCF(ABC):
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
