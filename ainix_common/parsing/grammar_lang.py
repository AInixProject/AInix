from arpeggio import ParserPython, visit_parse_tree, PTNodeVisitor, SemanticActionResults
from arpeggio import Optional, ZeroOrMore, Not, OneOrMore, EOF, ParserPython, \
    visit_parse_tree
from arpeggio import RegExMatch
from arpeggio.peg import PEGVisitor
import arpeggio.cleanpeg

# Lexical invariants
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


parser = ParserPython(peg_obj_parser, None)


def parse(string: str):
    return parser.parse(string)


class CustPegVisitor(PTNodeVisitor):
    def visit_ordered_choice(self, node, children: SemanticActionResults):
        print("VISIT ordered choice")
        raise NotImplemented

    def visit_sequence(self, node, children: SemanticActionResults):
        print("VISIT sequence", node)
        return 3


if __name__ == "__main__":
    ast = parse(r'Foo "b" Bar?')
    print(ast.__dir__())
    result = visit_parse_tree(ast, CustPegVisitor())
    print(result)
