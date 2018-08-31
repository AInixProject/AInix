from typing import List
import attr
import indexing.whooshbackend
from indexing.index import IndexBackendScheme, IndexBackendFields, IndexBackendABC
import parseast
from parseast import StringParser
from typecontext import TypeContext, AInixType
from whoosh.query import Term, And, Or
from whoosh.analysis.analyzers import Analyzer, StandardAnalyzer
from whoosh.analysis.tokenizers import RegexTokenizer
from whoosh.analysis.filters import LowercaseFilter


@attr.s(auto_attribs=True, frozen=True)
class Example:
    xquery: str
    ytext: str
    xtype: str
    ytype: str
    weight: float
    yindexable: str = None


class ExamplesIndex:
    """Provides a higher level interface around an IndexBackendABC specifically
    related to the domain of AInix examples"""
    DEFAULT_X_TYPE = "WordSequence"
    # TODO (DNGros): this shouldn't really be here. Should not depend on whoosh
    x_tokenizer = RegexTokenizer() | LowercaseFilter()

    def __init__(self, type_context: TypeContext, backend: IndexBackendABC = None):
        scheme = self.get_scheme()
        self.backend = backend if backend else \
            indexing.whooshbackend.WhooshIndexBackend(scheme)
        self.type_context = type_context

    @staticmethod
    def get_scheme() -> 'IndexBackendScheme':
        return IndexBackendScheme(
            xquery=IndexBackendFields.TEXT,
            ytext=IndexBackendFields.TEXT,
            xtype=IndexBackendFields.ID,
            ytype=IndexBackendFields.ID,
            yindexable=IndexBackendFields.SPACE_STORED_TEXT,
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
                new_example = Example(x, y, x_type, y_type, weight,
                                      self._get_yparsed_rep(y, y_type))
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

    def get_nearest_examples(
        self,
        x_value: str,
        choose_type_name: str = None,
        max_results = 10
    ) -> List[Example]:
        tokenized_x_value = (tok.text for tok in  self.x_tokenizer(x_value))
        query = Or([Term("xquery", term,) for term in tokenized_x_value])
        if choose_type_name:
            y_type_indexable_rep = parseast.indexable_repr_classify_type(choose_type_name)
            query &= Term("yindexable", y_type_indexable_rep)
        query_result = self.backend.query(query, max_results)
        list_result = [Example(**hit.doc) for hit in query_result]
        return list_result
