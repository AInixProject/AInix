from abc import ABC, abstractmethod
from parseast import ObjectChoiceNode


class StringTypeTranslateCF(ABC):
    """Translates a string to another type without taking any prior context
    into account (what the user or the system has said previously)"""
    # TODO (DNGros): add interface for batched training and prediction

    @abstractmethod
    def predict(self, x_string: str, y_type_name: str) -> ObjectChoiceNode:
        pass

    @abstractmethod
    def train(self, x_string: str, y_ast: ObjectChoiceNode):
        pass
