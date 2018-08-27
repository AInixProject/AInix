from parsing.typecontext import TypeContext, AInixType
import attr
from typing import List, Dict, Set
from collections import defaultdict
DEFAULT_X_TYPE = "WordSequence"


@attr.s(auto_attribs=True, frozen=True)
class TypePair:
    x_type: str
    y_type: str


@attr.s(auto_attribs=True, frozen=True)
class YString:
    value: str
    weight: float = 1


class ExampleContext:
    def __init__(self, type_context: TypeContext):
        self.type_context = type_context
        # examples go from a type pair -> dict going from x_string -> dict of string to weight
        self.examples: Dict[TypePair, Dict[str, Dict[str, float]]] = \
            defaultdict(dict)

