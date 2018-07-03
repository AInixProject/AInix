import attr
from pygments.token import Token
from pygments.lexers import BashLexer
import pygments
class BashParser():
    def __init__(self):
        self.lexer = BashLexer()

    def parse(self, text):
        lexed = list(pygments.lex(text, self.lexer))
        words = []
        error = False
        for tokenType, value in lexed:
            if tokenType in (Token.Text, Token.Literal.Number, Token.Name.Builtin, Token.Punctuation):
                stripped = value.strip()
                if len(stripped) > 0:
                    words.append(value)
            else:
                print("AAAHHHH", tokenType, value)
                error = True
                break
        return ParseResult(was_error = error, words = words)

@attr.s(frozen = True)
class ParseResult():
    words = attr.ib(converter=tuple)
    was_error = attr.ib(default=False)
    def get_first_word(self):
        return self.words[0]
