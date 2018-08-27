import indexing.index
from typing import Iterable, Dict
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.query import Term, And, Or


def _make_all_dict_values_strings(document: Dict):
    for key, value in document.items():
        document[key] = str(value)
    return document


class WhooshIndexBackend(indexing.index.IndexBackendABC):
    """An implementation of an index backend powered by Whoosh

    Args:
        scheme : the scheme to use for the index
    """
    def __init__(self, scheme: indexing.index.IndexBackendScheme):
        whoosh_scheme = self._convert_to_whoosh_scheme(scheme)
        self.index = create_in("indexdir", whoosh_scheme)
        self.searcher = None

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

    def add_documents(self, documents: Iterable[Dict]):
        # TODO (DNGros): look into setting proc field greater than 1
        # and generally making async
        with self.index.writer() as writer:
            for document in documents:
                _make_all_dict_values_strings(document)
                writer.add_document(**document)

    def field_or_terms(self, field_name, terms):
        whoosh_terms = [Term(field_name, term) for term in terms]
        query = Or(whoosh_terms, scale=0.95)
        # TODO (DNGros): figure out the contexts for a search object
        #with self.index.searcher() as searcher:
        #    result = searcher.search(query)
        self.searcher = self.index.searcher()
        result = self.searcher.search(query)
        # TODO (DNGros): Abstract away a return format for an IndexBackendABC
        return result

