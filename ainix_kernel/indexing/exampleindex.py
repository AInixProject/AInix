from typing import List, Generator, Dict, Tuple
import attr
import indexing.whooshbackend
from indexing.index import IndexBackendScheme, IndexBackendFields, IndexBackendABC
import parseast
from parseast import StringParser
from typecontext import TypeContext
from whoosh.query import Term, Or
from whoosh.analysis.tokenizers import RegexTokenizer
from whoosh.analysis.filters import LowercaseFilter
from indexing.examplestore import ExamplesStore, Example, \
    get_split_from_example, SPLIT_TYPE, DEFAULT_SPLITS, DataSplits


class ExamplesIndex(ExamplesStore):
    """An ExampleStore that also provides interfaces such as querying for nearest
    examples of a certain type.
    It effectively provides a higher level interface around an IndexBackendABC specifically
    related to the domain of AInix examples."""

    # TODO (DNGros): this tokenizer shouldn't really be here. Should not depend on whoosh
    x_tokenizer = RegexTokenizer() | LowercaseFilter()

    def __init__(self, type_context: TypeContext, backend: IndexBackendABC = None):
        super().__init__(type_context)
        scheme = self.get_scheme()
        self.backend = backend if backend else \
            indexing.whooshbackend.WhooshIndexBackend(scheme)

    @staticmethod
    def get_scheme() -> 'IndexBackendScheme':
        return IndexBackendScheme(
            xquery=IndexBackendFields.TEXT,
            ytext=IndexBackendFields.TEXT,
            xtype=IndexBackendFields.ID,
            ytype=IndexBackendFields.ID,
            yindexable=IndexBackendFields.SPACE_STORED_TEXT,
            weight=IndexBackendFields.TEXT,
            split=IndexBackendFields.KEYWORD
        )

    @staticmethod
    def get_default_ram_backend() -> 'IndexBackendABC':
        """Gets a default backend that does not touch any files and just
        keeps data in RAM"""
        return indexing.whooshbackend.WhooshIndexBackend(
            ExamplesIndex.get_scheme(), ram_only=True)

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
        splits: SPLIT_TYPE = DEFAULT_SPLITS
    ) -> None:
        for x in x_values:
            split = get_split_from_example(x, y_type, splits)
            for y, weight in zip(y_values, weights):
                new_example = Example(x, y, x_type, y_type, weight, split.value,
                                      self._get_yparsed_rep(y, y_type))
                self.add_example(new_example)

    def _dict_to_example(self, doc: Dict) -> Example:
        """Takes the dictionary form of an object and returns an example object"""
        return Example(**doc)

    def get_nearest_examples(
        self,
        x_value: str,
        choose_type_name: str = None,
        max_results = 10
    ) -> List[Example]:
        """
        Args:
            x_value: a string to look for the most similar example to
            choose_type_name: By optionally specifying this value you may require
                that a specific type choice appears in the example. You could for example
                only look for the nearest examples where the example features a choice
                between a Program type.
            max_results: The max number of examples to return

        Returns:
            A list of all examples that are potentially near the example, sorted
            in order where the 0th item is predicted to be nearest.
        """
        tokenized_x_value = (tok.text for tok in  self.x_tokenizer(x_value))
        query = Or([Term("xquery", term,) for term in tokenized_x_value])
        if choose_type_name:
            y_type_indexable_rep = parseast.indexable_repr_classify_type(choose_type_name)
            query &= Term("yindexable", y_type_indexable_rep)
        query_result = self.backend.query(query, max_results)
        list_result = [self._dict_to_example(hit.doc) for hit in query_result]
        return list_result

    def get_all_examples(
        self,
        filter_splits: Tuple[DataSplits, ...]=None
    ) -> Generator[Example, None, None]:
        """Yields all examples in the index"""
        yield from map(self._dict_to_example, self.backend.get_all_documents(filter_splits))


