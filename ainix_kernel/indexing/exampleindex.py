from typing import List, Generator, Dict, Tuple
import attr
from ainix_kernel.indexing.whooshbackend import WhooshIndexBackend
from ainix_kernel.indexing.index import IndexBackendScheme, IndexBackendFields, IndexBackendABC
from ainix_common.parsing import ast_components
from ainix_common.parsing.stringparser import StringParser
from ainix_common.parsing.typecontext import TypeContext
from whoosh.query import Term, Or, Every
from whoosh.analysis.tokenizers import RegexTokenizer
from whoosh.analysis.filters import LowercaseFilter
from ainix_kernel.indexing.examplestore import ExamplesStore, Example, \
    get_split_from_example, SPLIT_TYPE, DEFAULT_SPLITS, DataSplits
from ainix_common.util.strings import id_generator
import ainix_kernel.indexing.index
import copy


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
        self.parser = StringParser(type_context)
        self.backend = backend or WhooshIndexBackend(scheme)
        self.example_count = 0

    @staticmethod
    def get_scheme() -> 'IndexBackendScheme':
        return IndexBackendScheme(
            example_id=IndexBackendFields.NUMBER,
            xquery=IndexBackendFields.TEXT,
            ytext=IndexBackendFields.TEXT,
            xtype=IndexBackendFields.ID,
            ytype=IndexBackendFields.ID,
            yindexable=IndexBackendFields.SPACE_STORED_TEXT,
            y_set_id=IndexBackendFields.ID,
            weight=IndexBackendFields.ONE_INTERVAL_NUM,
            split=IndexBackendFields.KEYWORD
        )

    @staticmethod
    def get_default_ram_backend() -> 'IndexBackendABC':
        """Gets a default backend that does not
        if node is None:
        return Falsetouch any files and just
        keeps data in RAM"""
        return ainix_kernel.indexing.whooshbackend.WhooshIndexBackend(
            ExamplesIndex.get_scheme(), ram_only=True)

    def get_doc_count(self) -> int:
        return self.backend.get_doc_count()

    def _get_yparsed_rep(self, y_string: str, y_type: str) -> str:
        ast = self.parser.create_parse_tree(y_string, y_type)
        return ast.indexable_repr()

    def add_example(self, example: Example) -> None:
        self.backend.add_documents([attr.asdict(example)])
        self.example_count += 1

    def add_many_to_many_with_weighted(
        self,
        x_values: List[str],
        y_values: List[str],
        x_type: str,
        y_type: str,
        weights: List[float],
        splits: SPLIT_TYPE = DEFAULT_SPLITS
    ) -> None:
        y_group = id_generator(size=10)
        for x in x_values:
            split = get_split_from_example(x, y_type, splits)
            for y, weight in zip(y_values, weights):
                new_example = Example(self.example_count, x, y, x_type, y_type, weight, y_group,
                                      split=split.value,
                                      yindexable=self._get_yparsed_rep(y, y_type))
                self.add_example(new_example)

    def _dict_to_example(self, doc: Dict) -> Example:
        """Takes the dictionary form of an object and returns an example object"""
        # make a copy of the dict so we can mutate alter its keys without
        # mutating the input dict (this might be overkill....)
        doc_copy = copy.deepcopy(doc)
        doc_copy['weight'] = float(doc_copy['weight'])
        return Example(**doc_copy)

    def get_nearest_examples(
        self,
        x_value: str,
        choose_type_name: str = None,
        filter_splits=None,
        max_results=10
    ) -> Generator[Example, None, None]:
        """
        Args:
            filter_splits:
            x_value: a string to look for the most similar example to
            choose_type_name: By optionally specifying this value you may require
                that a specific type choice appears in the example. You could for example
                only look for the nearest examples where the example features a choice
                between a Program type.
            filter_splits: A tuple of DataSplits. If provided, only examples in one
                of the splits in the provided tuple will be returned
            max_results: The max number of examples to return

        Returns:
            A list of all examples that are potentially near the example, sorted
            in order where the 0th item is predicted to be nearest.
        """
        tokenized_x_value = (tok.text for tok in self.x_tokenizer(x_value))
        query = Or([Term("xquery", term,) for term in tokenized_x_value])
        if choose_type_name:
            y_type_indexable_rep = ast_components.indexable_repr_classify_type(choose_type_name)
            query &= Term("yindexable", y_type_indexable_rep)
        if filter_splits:
            query &= Or([Term("split", split.value) for split in filter_splits])
        query_result = self.backend.query(query, max_results)
        yield from (self._dict_to_example(hit.doc) for hit in query_result)

    def get_all_examples(
        self,
        filter_splits: Tuple[DataSplits, ...] = None
    ) -> Generator[Example, None, None]:
        """Yields all examples in the index"""
        if filter_splits is None or len(filter_splits) == 0:
            query = Every()
        else:
            query = Or([Term("split", split.value) for split in filter_splits])
        yield from (self._dict_to_example(hit.doc)
                    for hit in self.backend.query(query, max_results=None, score=False))

    def get_examples_from_y_set(self, y_set_id: str) -> List[Example]:
        query = Term("y_set_id", y_set_id)
        return [self._dict_to_example(hit.doc)
                for hit in self.backend.query(query, None, False)]

