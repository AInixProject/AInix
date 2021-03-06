from abc import ABC, abstractmethod
from typing import Iterable, Dict, List, Generator, Tuple
import enum
import whoosh.query
import whoosh.searching
import attr

from ainix_kernel.indexing.examplestore import DataSplits


# NOTE: for now just using actual whoosh query objects
#class Query(whoosh.query.Query):
#    """An query on IndexBackendj. For now to save time it will just exactly
#    copy a whoosh query. However, this should likely eventually be abstracted
#    away and tied to specific backend (or it might work to make other backends
#    just convert from whoosh to their query scheme. We'll figure it out later)"""
#    pass


# TODO (DNGros): figure out how want to do result and if need special object
# for it
#class Results(whoosh.searching.Results):
#    """Results from on IndexBackend. For now to save time it will just exactly
#    copy a whoosh query. However, this should likely eventually be abstracted
#    away and tied to specific backend"""
#    pass

@attr.s(auto_attribs=True)
class SearchHit:
    doc: Dict
    score: float = None


class IndexBackendFields(enum.Enum):
    TEXT = "TEXT_FIELD"
    ID = "ID_FILED"
    NUMBER = "NUMBER_FIELD"
    UNSTORED_TEXT = "UNSTORED_TEXT_FIELD"
    KEYWORD = "KEYWORD_FIELD"
    # A unstored texts that tokenizes purly on spaces
    SPACE_UNSTORED_TEXT = "UNSTORED_TEXT_FIELD_SPACE_TOKENIZE"
    SPACE_STORED_TEXT = "SPACE_FIELD_SPACE_TOKENIZE"
    # A ONE_INTERVAL_NUM is field in the interval [0, 1]
    ONE_INTERVAL_NUM = "ONE_INTERVAL_NUM_FIELD"


class IndexBackendScheme:
    def __init__(self, **kwargs: IndexBackendFields):
        self.fields: Dict[str, IndexBackendFields] = kwargs


class IndexBackendABC(ABC):
    """ABC for storing data and querying it. It offers more general
    purpose puts and query operations not specifically tied to AInix
    terminology. This is intended so we can experiment with different backends
    such as Whoosh or maybe eventually Lucene / its derivatives or maybe more
    vector based index scheme.

    Because I don't really know any better right now, it's interface is
    heavily based off whoosh's terminology."""
    @abstractmethod
    def add_documents(self, documents: Iterable[Dict]):
        pass

    @abstractmethod
    def get_doc_count(self):
        pass

    @abstractmethod
    def query(
        self,
        query: whoosh.query.Query,
        max_results: int = 10,
        score: bool = True
    ) -> List[SearchHit]:
        pass
