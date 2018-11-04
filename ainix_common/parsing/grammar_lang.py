"""This module is for handling parsers specified via a PEG-style grammar"""
from typing import Callable, Generator, Union, Tuple, Sequence

from arpeggio import ParserPython, visit_parse_tree, PTNodeVisitor, SemanticActionResults, \
    ParseTreeNode
from arpeggio import Optional, ZeroOrMore, Not, OneOrMore, EOF, ParserPython, \
    visit_parse_tree
from arpeggio import RegExMatch
from arpeggio.peg import PEGVisitor
import arpeggio.cleanpeg
from pyrsistent import v

from ainix_common.parsing import parse_primitives
from ainix_common.parsing.parse_primitives import ObjectParser, ObjectParserRun, \
    ObjectParserResult, ParseDelegationReturnMetadata, UnparseableObjectError, \
    ObjectParseFuncType, ArgParseDelegation, UnparsableTypeError
from ainix_common.parsing.typecontext import TypeContext, AInixObject

ASSIGNMENT = "="
ORDERED_CHOICE = "/"
ZERO_OR_MORE = "*"
ONE_OR_MORE = "+"
OPTIONAL = "?"
UNORDERED_GROUP = "#"
AND = "&"
NOT = "!"
OPEN = "("
CLOSE = ")"


# Define the grammar for parsing our grammars. How meta...
# Code adapted from https://github.com/igordejanovic/Arpeggio/blob/master/arpeggio/cleanpeg.py


#def peggrammar():
#    return OneOrMore(rule), EOF
#def rule():
#    return rule_name, ASSIGNMENT, ordered_choice
def peg_obj_parser(): return sequence, EOF


def ordered_choice(): return sequence, ZeroOrMore(ORDERED_CHOICE, sequence)


def sequence(): return OneOrMore(sufix)


#def prefix():
#    #return Optional([AND, NOT]), sufix
#    return sufix


def sufix():
    return expression, Optional(OPTIONAL)
                                 #ZERO_OR_MORE,
                                 #ONE_OR_MORE,
                                 #UNORDERED_GROUP])


def expression():
    return [
            # regex,
            arg_identifier,
            (OPEN, ordered_choice, CLOSE),
            str_match], Not(ASSIGNMENT)

# PEG Lexical rules
#def regex():
#    return [("r'", _(r'''[^'\\]*(?:\\.[^'\\]*)*'''), "'"),
#            ('r"', _(r'''[^"\\]*(?:\\.[^"\\]*)*'''), '"')]


#def rule_name():
#    return RegExMatch(r"[a-zA-Z_]([a-zA-Z_]|[0-9])*")


def arg_identifier():
    return RegExMatch(r"[a-zA-Z_]([a-zA-Z_]|[0-9])*")


def str_match():
    return RegExMatch(r'''(?s)('[^'\\]*(?:\\.[^'\\]*)*')|'''
                      r'''("[^"\\]*(?:\\.[^"\\]*)*")''')


def comment():
    return "//", RegExMatch(".*\n")


grammar_parser = ParserPython(peg_obj_parser, None)


def parse_grammar(string: str):
    return grammar_parser.parse(string)


# A type decription for the return type for non-terminal visitors of the PEG grammar
VisitorReturnType = Generator[
    ArgParseDelegation,
    ParseDelegationReturnMetadata,
    Tuple[ParseDelegationReturnMetadata, Sequence[ParseDelegationReturnMetadata]]
]


def _visit_str_match(node, string, left_offset) -> ParseDelegationReturnMetadata:
    """A grammar visitor for the literal string matches"""
    look_for = node.value[1:-1]
    does_start_with = string.startswith(look_for)
    if does_start_with:
        return ParseDelegationReturnMetadata(does_start_with, string, left_offset,
                                             node, len(look_for))
    else:
        return ParseDelegationReturnMetadata(False, string, left_offset, node, None)


def _visit_sufix(node, string, left_offset, run_data) -> VisitorReturnType:
    """A grammar visitor for the suffixes"""
    visitv = yield from gen_grammar_visitor(node[0], string, left_offset, run_data)
    expression, acceptables = visitv
    if len(node) > 1:
        sufix = node[1]
        if sufix == OPTIONAL:
            if not expression.parse_success:
                return ParseDelegationReturnMetadata.make_for_unparsed_string(string, None), v()
        else:
            raise ValueError("Unsupported sufix")
    return expression, acceptables


def gen_grammar_visitor(
    node: ParseTreeNode,
    string: str,
    left_offset: int,
    run_data: ObjectParserRun
) -> VisitorReturnType:
    """A custom visitor on the PEG grammer which does the delegation as necessary.
    It is a corroutine which yields delegations as needed. It returns metadata about
    whether or not it succeeded at each visit as well as a pvec of metadata which
    would need to be accepted as a valid parses assuming everything above it succeeds.

    This isn't a great description I know. If you are reading this and it unclear
    nag @DNGros to clean it up.
    """
    print("visiting", node, node.rule_name, "  :: ", string)
    if node.rule_name == "arg_identifier":
        slice_to_parse = (left_offset, left_offset+len(string))
        delegation = run_data.left_fill_arg(run_data.get_arg_by_name(node.value), slice_to_parse)
        parse_return = yield delegation
        if not parse_return.parse_success:
            parse_return = parse_return.add_fail(f"Stack Message: Fail on arg {node.value}")
        return parse_return, v(parse_return)
    elif node.rule_name == "str_match":
        return _visit_str_match(node, string, left_offset), v()
    elif node.rule_name == "sufix":
        out_return, things_to_accept = yield from _visit_sufix(node, string, left_offset, run_data)
        return out_return, things_to_accept
    else:
        remaining_string = string
        new_left_offset = left_offset
        acceptables = []
        if isinstance(node, arpeggio.NonTerminal):
            for child in node:
                visitv = yield from gen_grammar_visitor(
                    child, remaining_string, new_left_offset, run_data)
                parse_return, new_acceptables = visitv
                if not parse_return.parse_success:
                    print("FAIL on", child, string)
                    return parse_return, v()
                acceptables.extend(new_acceptables)
                remaining_string = parse_return.remaining_string
                new_left_offset += parse_return.remaining_right_starti
        # TODO (DNGros): Figure out what to put in as what_parsed here
        new_return = ParseDelegationReturnMetadata.create_from_substring(
            None, string, remaining_string, left_offset)
        return new_return, acceptables


def create_object_parser_from_grammar(
    type_context: TypeContext,
    parser_name: str,
    grammar: str,
    exclusive_type_name: str = None
) -> ObjectParser:
    return ObjectParser(
        type_context,
        parser_name ,
        _create_object_parser_func_from_grammar(grammar),
        _create_object_tostring_func_from_grammar(grammar),
        exclusive_type_name
    )


def _create_object_parser_func_from_grammar(
        grammar: str
) -> ObjectParseFuncType:
    grammar_ast = parse_grammar(grammar)

    def out_func(run_data: ObjectParserRun, string: str, result: ObjectParserResult):
        nonlocal grammar_ast
        visitv = yield from gen_grammar_visitor(grammar_ast, string, 0, run_data)
        parse_return, acceptables = visitv
        if not parse_return.parse_success:
            raise UnparseableObjectError(f"Error parseing string {string} with grammar {grammar}."
                                         f"Clunky 'stack trace' (can be made better): "
                                         f"{parse_return.fail_reason}")
        for acceptable_delegation in acceptables:
            result.accept_delegation(acceptable_delegation)
        result.remaining_start_i = parse_return.remaining_right_starti + \
                                   parse_return.original_start_offset

    return out_func


def _create_first_succeed_type_parser_func(
    ordered_implementations: Sequence[AInixObject]
):
    """Closure which creates a parser function which parses a bunch of implementations
    and takes the first one that succeeds"""
    def out_func(
        run: parse_primitives.TypeParserRun,
        string: str,
        result: parse_primitives.TypeParserResult
    ):
        slice_to_parse = (0, len(string))
        for impl in ordered_implementations:
            parse_return = yield run.delegate_parse_implementation(impl, slice_to_parse)
            if parse_return.parse_success:
                result.accept_delegation(parse_return)
        raise UnparsableTypeError("A first succeed parser did not find a valid implementation")

    return out_func


###################
## Unparsing
###################
class UnparseError(Exception):
    pass


def unparse_visitor(
    node: ParseTreeNode,
    result: parse_primitives.ObjectToStringResult,
    arg_map: parse_primitives.ObjectNodeArgMap
):
    if node.rule_name == "arg_identifier":
        _unparse_visit_identifier(node, result, arg_map)
    elif node.rule_name == "str_match":
        result.add_string(node.value[1:-1])
    elif node.rule_name == "sufix":
        _unparse_visit_suffix(node, result, arg_map)
    else:
        if isinstance(node, arpeggio.NonTerminal):
            for child in node:
                unparse_visitor(child, result, arg_map)


def _unparse_visit_identifier(
    node: ParseTreeNode,
    result: parse_primitives.ObjectToStringResult,
    arg_map: parse_primitives.ObjectNodeArgMap
):
    arg_name = node.value
    if not arg_map.is_argname_present(arg_name):
        raise UnparseError(f"Arg {arg_name} is not present")
    result.add_argname_tostring(arg_name)


def _unparse_visit_suffix(
    node: ParseTreeNode,
    result: parse_primitives.ObjectToStringResult,
    arg_map: parse_primitives.ObjectNodeArgMap
):
    snapshot = result.make_snapshot()
    try:
        unparse_visitor(node[0], result, arg_map)
    except UnparseError as e:
        suffix = node[1]
        if suffix == OPTIONAL:
            snapshot.restore()
        else:
            raise ValueError(f"Unimplemented suffix {suffix}")


def _create_object_tostring_func_from_grammar(
    grammar: str
) -> ObjectParseFuncType:
    grammar_ast = parse_grammar(grammar)

    def out_unparser(
        arg_map: parse_primitives.ObjectNodeArgMap,
        result: parse_primitives.ObjectToStringResult
    ):
        unparse_visitor(grammar_ast, result, arg_map)
    return out_unparser


