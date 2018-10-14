from typing import Callable

from arpeggio import ParserPython, visit_parse_tree, PTNodeVisitor, SemanticActionResults, \
    ParseTreeNode
from arpeggio import Optional, ZeroOrMore, Not, OneOrMore, EOF, ParserPython, \
    visit_parse_tree
from arpeggio import RegExMatch
from arpeggio.peg import PEGVisitor
import arpeggio.cleanpeg

# Lexical invariants
from ainix_common.parsing.parse_primitives import ObjectParser, ObjectParserRun, \
    ObjectParserResult, ArgParseDelegationReturn, UnparseableObjectError
from ainix_common.parsing.typecontext import TypeContext

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


# Test
grammar_parser = ParserPython(peg_obj_parser, None)
def parse_grammar(string: str):
    return grammar_parser.parse(string)


def gen_grammar_visitor(node: ParseTreeNode, string: str, run_data: ObjectParserRun):
    print("visiting", node, node.rule_name)
    if node.rule_name == "arg_identifier":
        parse_return = yield run_data.left_fill_arg(node.value, string)
        return parse_return
    else:
        remaining_string = string
        if isinstance(node, arpeggio.NonTerminal):
            for child in node:
                parse_return = yield from gen_grammar_visitor(child, remaining_string, run_data)
                if not parse_return.parse_success:
                    return parse_return
                remaining_string = parse_return.remaining_string
        return ArgParseDelegationReturn(True, remaining_string)


def _create_object_parser_func_from_grammar(
    grammar: str
) -> Callable[[ObjectParserRun, str, ObjectParserResult], None]:
    grammar_ast = parse_grammar(grammar)

    def out_func(run_data: ObjectParserRun, string: str, result: ObjectParserResult):
        nonlocal grammar_ast
        parse_return = yield from gen_grammar_visitor(grammar_ast, string, run_data)
        if not parse_return.parse_success:
            raise UnparseableObjectError(f"Error parseing string {string} with grammar {grammar}")
        yield

    return out_func


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
        exclusive_type_name
    )


class CustPegVisitor(PTNodeVisitor):
    def visit_ordered_choice(self, node, children: SemanticActionResults):
        print("VISIT ordered choice")
        raise NotImplemented

    def visit_sequence(self, node, children: SemanticActionResults):
        print("VISIT sequence", node)
        return 3


if __name__ == "__main__":
    ast = parse_grammar(r'Foo "b" Bar?')
    print(ast.__dir__())
    #result = visit_parse_tree(ast, CustPegVisitor())
    #print(result)
    print(ast.name)
    for child in ast:
        print(child.name)

    #import inspect
    #import types
    #def bar():
    #    yield "b"
    #    return 5


    #def foo():
    #    b = bar()
    #    if isinstance(b, types.GeneratorType):
    #        print("yield from")
    #        w = yield from b
    #        print("w = ", w)
    #    else:
    #        yield b
    #    print("Done")
    #    return 3

    #g = foo()
    #while True:
    #    try:
    #        print(next(g))
    #    except StopIteration as se:
    #        print("end", se.value + 2)
    #        break

