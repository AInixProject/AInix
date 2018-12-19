"""Methods for populating a TypeContext off of *.ainix.yaml files which
defines types, objects, and parsers. The file may also contain examples,
but the loaders in this module just ignores those. If you want to load
examples use ainix_kenel.indexing.exampleloader"""
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from typing import Dict, List, Optional, Callable, IO
from ainix_common.parsing import typecontext
from ainix_common.parsing import parse_primitives
from ainix_common.parsing import examplecontext
import importlib.util
from ainix_common.parsing import grammar_lang
import os


def load_path(
    path : str,
    type_context: typecontext.TypeContext,
    up_search_limit: int = 0
) -> None:
    """Loads a *.ainix.yaml file and registers and defined types
    or objects with the supplied type_context"""
    # TODO (DNGros): validate that is actually a *.ainix.yaml file
    # TODO (DNGros): allow for you to specify a  path and recurse down
    load_root = os.path.dirname(path)
    new_path = path
    for up_search_i in range(up_search_limit+1):
        if os.path.isfile(new_path):
            break
        load_root = os.path.join(os.path.pardir, load_root)
        new_path = os.path.join(os.path.pardir, new_path)
    else:
        raise ValueError(f"Path {path} not found in working dir or"
                         f"{up_search_limit} paths above")

    with open(new_path, 'r') as f:
        load_yaml(f, type_context, load_root)


def load_yaml(
    filelike: IO,
    type_context: typecontext.TypeContext,
    load_root=None
) -> None:
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
        type_context: type context we are loading into.
        load_root: the directory that is loading from. Used for relative references
    """
    for define in parsed_doc['defines']:
        what_to_define = define['define_new']
        if what_to_define == "type":
            _load_type(define, type_context)
        elif what_to_define == "object":
            _load_object(define, type_context)
        elif what_to_define == "type_parser":
            _load_type_parser(define, type_context, load_root)
        elif what_to_define == "object_parser":
            _load_object_parser(define, type_context, load_root)
        elif what_to_define == "example_set":
            pass
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


def _load_type(define, type_context: typecontext.TypeContext) -> None:
    """Parses the serialized form of a type and adds it to the supplied TypeContext"""
    # TODO (DNGros): validate that not extra keys in the define (would help catch people's typos)
    typecontext.AInixType(
        type_context,
        define['name'],
        default_type_parser_name=define.get("default_type_parser"),
        default_object_parser_name=define.get("default_object_parser"),
        allowed_attributes=define.get("allowed_attributes")
    )


def _load_object(define, type_context: typecontext.TypeContext):
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


def _extract_parse_func_from_py_module(name: str, module_name: str, load_root: str) -> Callable:
    """Extracts a parse function when the source of the function a python module.
    It can also be used for extracting a to_string func"""
    file_name = os.path.join(load_root, module_name)
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


def _create_type_parser_from_python_module(
    type_context: typecontext.TypeContext,
    load_root: str,
    type_name: Optional[str],
    parser_name: str,
    to_string_name: Optional[str],
    module_name_to_load_from: str
):
    """Creates a type parser based off python code"""
    parse_func = _extract_parse_func_from_py_module(
        parser_name, module_name_to_load_from, load_root)
    if to_string_name:
        to_string_func = _extract_parse_func_from_py_module(
            to_string_name, module_name_to_load_from, load_root)
    else:
        to_string_func = None
    parse_primitives.TypeParser(
        type_context, parser_name=parser_name,
        parse_function=parse_func,
        to_string_function=to_string_func,
        type_name=type_name,
    )


def _load_type_parser(
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
    source = define.get("source", "python_module")
    if source == "python_module":
        return _create_type_parser_from_python_module(
            type_context=type_context,
            load_root=load_root,
            type_name=define.get('type'),
            parser_name=define['name'],
            to_string_name=define.get('to_string_func', None),
            module_name_to_load_from=define['file']
        )
    else:
        raise ValueError(f"Unrecognized source {source} for TypeParser "
                         f"{define['name']}")


def _create_object_parser_from_python_module(
    type_context: typecontext.TypeContext,
    load_root: str,
    parser_name: str,
    to_string_name: str,
    module_name_to_load_from: str,
    exclusive_type_name: Optional[str]
):
    """Creates a type parser based off python code"""
    parse_func = _extract_parse_func_from_py_module(
        parser_name, module_name_to_load_from, load_root)
    if to_string_name:
        to_string_func = _extract_parse_func_from_py_module(
            to_string_name, module_name_to_load_from, load_root)
    else:
        to_string_func = None
    parse_primitives.ObjectParser(
        type_context,
        parser_name=parser_name,
        parse_function=parse_func,
        to_string_function=to_string_func,
        exclusive_type_name=exclusive_type_name
    )

def _load_object_parser(
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
    source = define.get("source", "python_module")
    if source == "python_module":
        _create_object_parser_from_python_module(
            type_context=type_context,
            load_root=load_root,
            parser_name=define['name'],
            to_string_name=define.get('to_string_func', None),
            module_name_to_load_from=define['file'],
            exclusive_type_name=define.get('type')
        )
    elif source == "arg_grammar":
        grammar_lang.create_object_parser_from_grammar(
            type_context=type_context,
            parser_name=define['name'],
            grammar=define['grammar'],
            exclusive_type_name=define.get('type')
        )
    else:
        raise ValueError(f"Unrecognized source {source} for TypeParser "
                         f"{define['name']}")


