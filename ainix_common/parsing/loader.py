import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from typing import Dict, List, Optional
import typecontext


def load_path(path : str, type_context: typecontext.TypeContext) -> None:
    """Loads a *.ainix.yaml file and registers and defined types
    or objects with the supplied type_context"""
    # TODO (DNGros): validate that is actually a *.ainix.yaml file
    # TODO (DNGros): allow for you to specify a  path and recurse down
    with open(path, 'r') as f:
        load_yaml(f, type_context)


def load_yaml(filelike, type_context: typecontext.TypeContext) -> None:
    doc = yaml.safe_load(filelike)
    _load(doc, type_context)


def _load(parsed_doc : Dict, type_context: typecontext.TypeContext) -> None:
    """Same as load_path() but accepts an already parsed dict.
    This done as seperate method incase we wish to support
    other serialization data formats in the future."""
    for define in parsed_doc['defines']:
        what_to_define = define['define_new']
        if what_to_define == "type":
            _parse_type(define, type_context)
        elif what_to_define == "object_name":
            _parse_object(define, type_context)
        else:
            raise ValueError("Unrecognized define_new value " + str(what_to_define))


def _parse_argument(doc: Dict, type_context: typecontext.TypeContext) -> Optional[typecontext.AInixArgument]:
    """Parses the a serialized form of an argument"""
    if doc is None:
        return None
    return typecontext.AInixArgument(
        type_context,
        doc['name'],
        type_name=doc['type'],
        type_parser_name=doc.get("type_parser"),
        required=doc.get("required", False),
        arg_data=doc.get("arg_data")
    )


def _parse_arguments(
    doc: List,
    type_context: typecontext.TypeContext
) -> List[typecontext.AInixArgument]:
    """Parses the serialized form of a list of arguments"""
    if doc is None:
        return []
    return [_parse_argument(arg_doc, type_context) for arg_doc in doc]


def _parse_type(define, type_context: typecontext.TypeContext) -> None:
    """Parses the serialized form of a type and adds it to the supplied TypeContext"""
    # TODO (DNGros): validate that not extra keys in the define (would help catch people's typos)
    typecontext.AInixType(
        type_context,
        define['name'],
        default_type_parser_name=define.get("default_type_parser"),
        default_object_parser_name=define.get("default_object_parser"),
        allowed_attributes=define.get("allowed_attributes")
    )


def _parse_object(define, type_context: typecontext.TypeContext):
    """Parses the serialized form of a object_name and adds it to the supplied TypeContext"""
    # TODO (DNGros): validate that not extra keys in the define (would help catch people's typos)
    typecontext.AInixObject(
        type_context,
        name=define['name'],
        type_name=define.get("type"),
        children=_parse_arguments(define.get("children"), type_context),
        direct_sibling=_parse_argument(define.get("direct_sibling"), type_context),
        preferred_object_parser_name=define.get("preferred_object_parser"),
        type_data=define.get("type_data")  # TODO: validate is a dict
    )
