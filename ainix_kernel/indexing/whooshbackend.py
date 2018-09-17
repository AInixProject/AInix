import ainix_kernel.indexing.index
from typing import Iterable, Dict, List, Optional, Generator, Tuple
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.searching import Searcher, Results
from whoosh.analysis import KeywordAnalyzer
import whoosh.query
from whoosh.query import Or, Term
import os
from indexing.examplestore import DataSplits


def _make_all_dict_values_strings(document: Dict):
    for key, value in document.items():
        document[key] = str(value)
    return document


def _convert_whoosh_result_to_our_result(
        whoosh_result: Results
) -> Generator[ainix_kernel.indexing.index.SearchHit, None, None]:
    yield from (ainix_kernel.indexing.index.SearchHit(result.fields(), result.score)
                for result in whoosh_result)


def _create_ram_index(schema):
    from whoosh.filedb.filestore import RamStorage
    from whoosh.index import FileIndex
    storage = RamStorage()
    return FileIndex.create(storage, schema)


class WhooshIndexBackend(ainix_kernel.indexing.index.IndexBackendABC):
    """An implementation of an index backend powered by Whoosh

    Args:
        scheme : the scheme to use for the index
        ram_only : keeps the index in ram. Useful for testing without polluting filesystem
    """
    INDEX_DIR = "indexdir"

    def __init__(self, scheme: ainix_kernel.indexing.index.IndexBackendScheme,
                 ram_only: bool = False):
        whoosh_scheme = self._convert_to_whoosh_scheme(scheme)
        if not ram_only:
            if not os.path.exists(self.INDEX_DIR):
                os.makedirs(self.INDEX_DIR)
            self.index = create_in(self.INDEX_DIR, whoosh_scheme)
        else:
            self.index = _create_ram_index(whoosh_scheme)
        self.searcher: Optional[Searcher] = None

    @staticmethod
    def _convert_to_whoosh_field(field: indexing.index.IndexBackendFields) -> FieldType:
        """
        Args:
            field: the IndexBackendField we want to convert

        Returns:
            the Whoosh form of that field
        """
        if field == ainix_kernel.indexing.index.IndexBackendFields.TEXT:
            return TEXT(stored=True)
        elif field == ainix_kernel.indexing.index.IndexBackendFields.ID:
            return ID(stored=True)
        elif field == ainix_kernel.indexing.index.IndexBackendFields.NUMBER:
            return NUMERIC(stored=True)
        elif field == ainix_kernel.indexing.index.IndexBackendFields.UNSTORED_TEXT:
            return TEXT(stored=False)
        elif field == ainix_kernel.indexing.index.IndexBackendFields.SPACE_UNSTORED_TEXT:
            return TEXT(stored=False, analyzer=KeywordAnalyzer())
        elif field == ainix_kernel.indexing.index.IndexBackendFields.SPACE_STORED_TEXT:
            return TEXT(stored=True, analyzer=KeywordAnalyzer())
        elif field == ainix_kernel.indexing.index.IndexBackendFields.KEYWORD:
            return KEYWORD(stored=True)
        elif field == ainix_kernel.indexing.index.IndexBackendFields.ONE_INTERVAL_NUM:
            return NUMERIC(float, stored=True)
        else:
            raise ValueError(
                f"WhooshIndexBackend does not support {field} fields.")

    @staticmethod
    def _convert_to_whoosh_scheme(
            scheme: ainix_kernel.indexing.index.IndexBackendScheme) -> Schema:
        """Converts our representation of our a scheme to whoosh's"""
        whoosh_scheme = Schema()
        for name, field in scheme.fields.items():
            whoosh_field = WhooshIndexBackend._convert_to_whoosh_field(field)
            whoosh_scheme.add(name, whoosh_field)
        return whoosh_scheme

    def _reset_searcher(self):
        if self.searcher is not None:
            self.searcher.close()
            self.searcher = None

    def _set_searcher(self):
        if self.searcher is None:
            self.searcher = self.index.searcher()

    def add_documents(self, documents: Iterable[Dict]):
        # TODO (DNGros): look into setting proc field greater than 1
        # and generally making async

        # When docs change, we should close any searchers
        self._reset_searcher()
        # write it
        with self.index.writer() as writer:
            for document in documents:
                _make_all_dict_values_strings(document)
                writer.add_document(**document)

    def query(
        self,
        query: whoosh.query.Query,
        max_results: Optional[int] = 10,
        score: bool = True
    ) -> Generator[ainix_kernel.indexing.index.SearchHit, None, None]:
        # TODO (DNGros): figure out the contexts for a search object and when
        # to close it and stuff
        self._set_searcher()
        result = self.searcher.search(query, limit=max_results, scored=score)
        yield from _convert_whoosh_result_to_our_result(result)

