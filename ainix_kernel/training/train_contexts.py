"""Quirky file for hardcoding some stuff which really shouldn't be hardcoded"""
import ainix_kernel.indexing.exampleindex
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing import exampleloader
from ainix_kernel.indexing.exampleindex import ExamplesIndex
from ainix_kernel.indexing.examplestore import ExamplesStore
import os

ALL_EXAMPLE_NAMES = ("numbers", "pwd", "ls", "cat", "head", "cp", "wc",
                     "mkdir", "echo", "mv", "touch")


def load_all_examples(tc: TypeContext) -> ExamplesStore:
    print("""Loading all examples...""")
    dirname, filename = os.path.split(os.path.abspath(__file__))
    index = ainix_kernel.indexing.exampleindex.ExamplesIndex(tc)
    for f in ALL_EXAMPLE_NAMES:
        exampleloader.load_path(f"{dirname}/../../builtin_types/{f}_examples.ainix.yaml", index)
    return index
