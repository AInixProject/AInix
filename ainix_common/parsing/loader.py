import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from typing import Dict, List, Optional
from typegraph import TypeGraph, AInixArgument


def load(path, type_graph : TypeGraph) -> None:
    """Loads a *.ainix.yaml file and registers and defined types
    or objects with the supplied type_graph"""
    with open(path, 'r') as f:
        doc = yaml.safe_load(f)
    _load(doc, type_graph)


def _load(parsed_doc : Dict, type_graph : TypeGraph) -> None:
    """Same as load() but accepts an already parsed dict.
    This done as seperate method incase we wish to support
    other serialization data formats in the future."""
    for define in parsed_doc['defines']:
        what_to_define = define['define_new']
        if what_to_define == "type":
            _parse_type(define, type_graph)
        elif what_to_define == "object":
            _parse_object(define, type_graph)
        else:
            raise ValueError("Unrecognized define_new value " + str(what_to_define))


def _parse_argument(doc : Dict) -> Optional[AInixArgument]:
    """Parses the a serialized form of an argument"""
    if doc is None:
        return None
    return AInixArgument(
        name = doc['name'],
        type = doc['type'],
        type_parser=doc.get("type_parser"),
        required=doc.get("required",False),
        arg_data=doc.get("arg_data")
    )


def _parse_arguments(doc : List) -> List[AInixArgument]:
    """Parses the serialized form of a list of arguments"""
    if doc is None:
        return []
    return [_parse_argument(arg_doc) for arg_doc in doc]


def _parse_type(define, type_graph) -> None:
    """Parses the serialized form of a type and adds it to the supplied TypeGraph"""
    # TODO (DNGros): validate that not extra keys in the define (would help catch people's typos)
    type_graph.create_type(
        name=define['name'],
        default_object_parser=define.get("default_object_parser"),
        default_type_parser=define.get("default_type_parser"),
        allowed_attributes=define.get("allowed_attributes")
    )


def _parse_object(define, type_graph):
    """Parses the serialized form of a object and adds it to the supplied TypeGraph"""
    # TODO (DNGros): validate that not extra keys in the define (would help catch people's typos)
    type_graph.create_object(
        name=define['name'],
        type=define.get("type"),
        children=_parse_arguments(define.get("children")),
        direct_sibling=_parse_argument(define.get("direct_sibling")),
        type_data=define.get("type_data")  # TODO: validate is a dict
    )
