from abc import ABC, abstractmethod
import attr
from typing import Iterable, Dict, List
from ainix_common.parsing import examplecontext
import indexing.whooshbackend
import enum
import attr
from typecontext import TypeContext, AInixType, AInixArgument
from parseast import StringParser


@attr.s(auto_attribs=True, frozen=True)
class Example:
    xquery: str
    ytext: str
    xtype: str
    ytype: str
    yparsed_rep: str
    weight: float


class ExamplesIndex:
    """Provides a higher level interface around an IndexBackendABC specifically
    related to the domain of AInix examples"""
    DEFAULT_X_TYPE = "WordSequence"

    def __init__(self, type_context: TypeContext):
        scheme = self._create_scheme()
        self.backend = indexing.whooshbackend.WhooshIndexBackend(scheme)
        self.type_context = type_context

    @staticmethod
    def _create_scheme() -> 'IndexBackendScheme':
        return IndexBackendScheme(
            xquery=IndexBackendFields.TEXT,
            ytext=IndexBackendFields.TEXT,
            xtype=IndexBackendFields.ID,
            ytype=IndexBackendFields.ID,
            yparsed_rep=IndexBackendFields.TEXT,
            weight=IndexBackendFields.TEXT
        )

    def _get_yparsed_rep(self, y_string: str, y_type: str) -> str:
        parser = StringParser(self.type_context.get_type_by_name(y_type))
        # TODO (DNGros): cache the parsers for each type
        ast = parser.create_parse_tree(y_string)
        return ast.indexable_repr()

    def add_example(self, example: Example) -> None:
        self.backend.add_documents([attr.asdict(example)])

    def add_many_to_many_with_weighted(
            self,
            x_values: List[str],
            y_values: List[str],
            x_type: str,
            y_type: str,
            weights: List[float],
    ) -> None:
        for x in x_values:
            for y, weight in zip(y_values, weights):
                new_example = Example(x, y, x_type, y_type,
                                      self._get_yparsed_rep(y), weight)
                self.add_example(new_example)

    def _default_weight(self, i: int, n: int):
        """Gets a default weight for a value. Each value in the sequence
        is half as preferable as the one before it

        Args:
            i : index in the sequence of values (zero indexed)
            n : total number of values in sequence
        """
        if i+1 > n:
            raise ValueError()
        sequence_sum = 2**n-1
        return (2**(n-i-1))/sequence_sum

    def add_many_to_many_default_weight(
        self,
        x_values: List[str],
        y_values: List[str],
        x_type: str,
        y_type: str
    ) -> None:
        """Adds several examples with the y_values default weighted."""
        y_count = len(y_values)
        weights = [self._default_weight(i, y_count)
                   for i, y in enumerate(y_values)]
        self.add_many_to_many_with_weighted(x_values, y_values,
                                            x_type, y_type, weights)


class IndexBackendFields(enum.Enum):
    TEXT = "TEXT_FIELD"
    ID = "ID_FILED"
    NUMBER = "NUMBER_FIELD"
    UNSTORED_TEXT = "UNSTORED_TEXT_FIELD"


class IndexBackendScheme:
    def __init__(self, **kwargs: IndexBackendFields):
        self.fields = kwargs


class IndexBackendABC(ABC):
    """ABC for storing data and querying it. It offers more general
    purpose puts and query operations not specifically tied to AInix
    terminology. This is intended so we can experiment with different backends
    such as Whoosh or maybe eventually Lucene / its derivatives.

    Because I don't really know any better right now, it's interface is
    heavily based off whoosh's terminology."""
    @abstractmethod
    def add_documents(self, documents: Iterable[Dict]):
        pass
