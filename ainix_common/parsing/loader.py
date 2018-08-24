import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from typing import Dict, List, Optional, Callable, IO
import typecontext
import parse_primitives
import importlib.util
import os


def load_path(path : str, type_context: typecontext.TypeContext) -> None:
    """Loads a *.ainix.yaml file and registers and defined types
    or objects with the supplied type_context"""
    # TODO (DNGros): validate that is actually a *.ainix.yaml file
    # TODO (DNGros): allow for you to specify a  path and recurse down
    load_root = os.path.dirname(path)
    with open(path, 'r') as f:
        load_yaml(f, type_context, load_root)


def load_yaml(filelike: IO, type_context: typecontext.TypeContext, load_root=None) -> None:
    doc = yaml.safe_load(filelike)
    _load(doc, type_context, load_root)


def _load(
    parsed_doc: Dict,
    type_context: typecontext.TypeContext,
    load_root: str
) -> None:
    """Same as load_path() but accepts an already parsed dict.
    This done as seperate method incase we wish to support
    other serialization data formats in the future.

    Args:
        parsed_doc: dict form of file we are loading from
        type_context: type context we are loading into
        load_root: the directory that is loading from. Used for relative references
    """
    for define in parsed_doc['defines']:
        what_to_define = define['define_new']
        if what_to_define == "type":
            _parse_type(define, type_context)
        elif what_to_define == "object":
            _parse_object(define, type_context)
        elif what_to_define == "type_parser":
            _parse_type_parser(define, type_context, load_root)
        elif what_to_define == "object_parser":
            _parse_object_parser(define, type_context, load_root)
        else:
            raise ValueError(f"Unrecognized define_new value {what_to_define}")


def _parse_argument(
    doc: Dict,
    type_context: typecontext.TypeContext
) -> Optional[typecontext.AInixArgument]:
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
        preferred_object_parser_name=define.get("preferred_object_parser"),
        type_data=define.get("type_data")  # TODO: validate is a dict
    )


def _extract_parse_func_from_py_module(define: Dict, load_root: str) -> Callable:
    """Extracts a parse function when the source of the function a python module"""
    file_name = os.path.join(load_root, define['file'])
    name = define['name']
    parser_module_spec = importlib.util.spec_from_file_location(
        "imparser." + name, file_name)
    parser_module = importlib.util.module_from_spec(parser_module_spec)
    parser_module_spec.loader.exec_module(parser_module)
    if name not in parser_module.__dict__:
        raise ValueError(f"Unable to find parse function {name} in {file_name}")
    parse_func = parser_module.__dict__[name]
    if not callable(parse_func):
        raise ValueError(f"Expected parse function of {name} in "
                         f"{file_name} to be callable")
    return parse_func


def _extract_parse_function(define: Dict, load_root: str) -> Callable:
    """Parses a define and gets out a parse_function for a parser. Currently
    this function works for both TypeParsers and ObjectParsers but this might
    change in the future if more sources are added."""
    source = define.get("source", "python_module")
    if source == "python_module":
        return _extract_parse_func_from_py_module(define, load_root)
    else:
        raise ValueError(f"Unrecognized source {source} for TypeParser "
                         f"{define['name']}")


def _parse_type_parser(
    define: Dict,
    type_context: typecontext.TypeContext,
    load_root: str
) -> None:
    """
    Args:
        define: the serialized form of the define of a new TypeParser
        type_context: the context to define the parser in.
        load_root: path to use the root for loading
    """
    parse_func = _extract_parse_function(define, load_root)
    parse_primitives.TypeParser(
        type_context, parser_name=define['name'],
        parse_function=parse_func,
        type_name=define.get('type')
    )


def _parse_object_parser(
    define: Dict,
    type_context: typecontext.TypeContext,
    load_root: str
) -> None:
    """
    Args:
        define: the serialized form of the define of a new ObjectParser
        type_context: the context to define the parser in.
        load_root: The root load context. Used as the relative start point for
            any file imports
    """
    parse_func = _extract_parse_function(define, load_root)
    parse_primitives.ObjectParser(
        type_context,
        parser_name=define['name'],
        parse_function=parse_func,
        type_name=define.get('type')
    )
