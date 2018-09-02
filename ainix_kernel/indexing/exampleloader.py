"""This module provides methods for loading examples and tasks from
an *.ainix.yaml file. This module is not for loading types or parsers.
For that use ainix_common.parsing.loader"""
# TODO (DNGros): this kinda repeats code from the type loader. Figure out
# how to make this DRYer...
import yaml
from indexing.examplestore import ExamplesStore
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from typing import Dict, IO


def load_path(
    path : str,
    index: ExamplesStore,
) -> None:
    """Loads a *.ainix.yaml file and registers and definesgT
    or objects with the supplied type_context"""
    # TODO (DNGros): validate that is actually a *.ainix.yaml file
    # TODO (DNGros): allow for you to specify a path and recurse down
    with open(path, 'r') as f:
        load_yaml(f, index)


def load_yaml(filelike: IO, index: ExamplesStore) -> None:
    doc = yaml.safe_load(filelike)
    _load(doc, index)


def _load(
    parsed_doc: Dict,
    index: ExamplesStore,
) -> None:
    """
    Args:
        parsed_doc: dict form of file we are loading from
        type_context: type context we are loading into.
        load_root: the directory that is loading from. Used for relative references
    """
    for define in parsed_doc['defines']:
        what_to_define = define['define_new']
        if what_to_define == "type":
            pass
        elif what_to_define == "object":
            pass
        elif what_to_define == "type_parser":
            pass
        elif what_to_define == "object_parser":
            pass
        elif what_to_define == "example_set":
            _load_example_set(define, index)
        else:
            raise ValueError(f"Unrecognized define_new value {what_to_define}")


def _load_single_example(
    example_dict: Dict,
    xtype: str,
    ytype: str,
    load_index: ExamplesStore
):
    x = example_dict['x']
    if not isinstance(x, list):
        x = [x]
    y = example_dict['y']
    if not isinstance(y, list):
        y = [y]
    x = list(map(str, x))
    y = list(map(str, y))
    load_index.add_many_to_many_default_weight(x, y, xtype, ytype)


def _load_example_set(define: Dict, load_index: ExamplesStore):
    y_type = define['y_type']
    x_type = define.get('x_type', load_index.DEFAULT_X_TYPE)
    examples = define['examples']
    for example in examples:
        _load_single_example(example, x_type, y_type, load_index)
