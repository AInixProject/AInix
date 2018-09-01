from models.SeaCR.seacr import Searcher
from indexing.exampleindex import ExamplesIndex
from ainix_common.parsing.typecontext import TypeContext


class SimpleSearcher(Searcher):
    def __init__(self, index: ExamplesIndex):
        self.index = index

    def query_string(self, x_string: str, y_type: str):
        pass
