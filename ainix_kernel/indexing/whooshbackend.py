import indexing.index
from typing import Iterable, Dict, List, Optional
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.searching import Searcher, Results
from whoosh.analysis import KeywordAnalyzer


def _make_all_dict_values_strings(document: Dict):
    for key, value in document.items():
        document[key] = str(value)
    return document


def _convert_whoosh_result_to_our_result(
        whoosh_result: Results
) -> List[indexing.index.SearchHit]:
    return [indexing.index.SearchHit(result.fields(), result.score)
            for result in whoosh_result]


def _create_ram_index(schema):
    from whoosh.filedb.filestore import RamStorage
    from whoosh.index import FileIndex
    storage = RamStorage()
    return FileIndex.create(storage, schema)


class WhooshIndexBackend(indexing.index.IndexBackendABC):
    """An implementation of an index backend powered by Whoosh

    Args:
        scheme : the scheme to use for the index
        ram_only : keeps the index in ram. Useful for testing without polluting filesystem
    """
    def __init__(self, scheme: indexing.index.IndexBackendScheme, ram_only: bool = False):
        whoosh_scheme = self._convert_to_whoosh_scheme(scheme)
        if not ram_only:
            self.index = create_in("indexdir", whoosh_scheme)
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
        if field == indexing.index.IndexBackendFields.TEXT:
            return TEXT(stored=True)
        elif field == indexing.index.IndexBackendFields.ID:
            return ID(stored=True)
        elif field == indexing.index.IndexBackendFields.NUMBER:
            return NUMERIC(stored=True)
        elif field == indexing.index.IndexBackendFields.UNSTORED_TEXT:
            return TEXT(stored=False)
        elif field == indexing.index.IndexBackendFields.SPACE_UNSTORED_TEXT:
            return TEXT(stored=False, analyzer=KeywordAnalyzer())
        else:
            raise ValueError(
                f"WhooshIndexBackend does not support {field} fields.")

    @staticmethod
    def _convert_to_whoosh_scheme(scheme: indexing.index.IndexBackendScheme) -> Schema:
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

    def query(self, query: indexing.index.Query) -> List[indexing.index.SearchHit]:
        # TODO (DNGros): figure out the contexts for a search object and when
        # to close it and stuff
        if self.searcher is None:
            self.searcher = self.index.searcher()
        result = self.searcher.search(query)
        return _convert_whoosh_result_to_our_result(result)


