"""Quirky file for hardcoding some stuff which really shouldn't be hardcoded"""
import ainix_kernel.indexing.exampleindex
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing import exampleloader
from ainix_kernel.indexing.exampleindex import ExamplesIndex
from ainix_kernel.indexing.examplestore import ExamplesStore, SPLIT_PROPORTIONS_TYPE, \
    DEFAULT_SPLITS, DataSplitter
import os

ALL_EXAMPLE_NAMES = (
    "find", "numbers",
    "pwd", "ls", "cat", "head", "cp", "wc", "cd",
    "mkdir", "echo", "mv", "touch", "sleep", "split",
    'tar'
    )


def load_all_examples(
    tc: TypeContext,
    splits: SPLIT_PROPORTIONS_TYPE = DEFAULT_SPLITS,
    split_seed: int = None
) -> ExamplesStore:
    splitter = DataSplitter(splits, split_seed)
    dirname, filename = os.path.split(os.path.abspath(__file__))
    index = ainix_kernel.indexing.examplestore.BasicExampleStore(tc)
    for f in ALL_EXAMPLE_NAMES:
        if f in ("numbers",):
            continue
        exampleloader.load_path(f"{dirname}/../../builtin_types/{f}_examples.ainix.yaml",
                                index, splitter)
    return index


def load_tellia_examples(tc: TypeContext) -> ExamplesStore:
    dirname, filename = os.path.split(os.path.abspath(__file__))
    index = ainix_kernel.indexing.examplestore.BasicExampleStore(tc)
    exampleloader.load_path(
        f"{dirname}/../../builtin_types/otherdata/tellina/tellina.ainix.yaml", index)
    return index


def load_all_and_tellina(tc: TypeContext) -> ExamplesStore:
    dirname, filename = os.path.split(os.path.abspath(__file__))
    index = ainix_kernel.indexing.examplestore.BasicExampleStore(tc)
    for f in ALL_EXAMPLE_NAMES:
        if f in ("numbers",):
            continue
        exampleloader.load_path(f"{dirname}/../../builtin_types/{f}_examples.ainix.yaml", index)
    exampleloader.load_path(
        f"{dirname}/../../builtin_types/otherdata/tellina/tellina.ainix.yaml", index)
    return index
