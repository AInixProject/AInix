from typing import List

import attr
from pygments.token import Token
from pygments.lexers import BashLexer
import pygments
class BashParser():
    def __init__(self):
        self.lexer = BashLexer()

    def parse(self, text):
        lexed = list(pygments.lex(text, self.lexer))
        words: List[str] = []
        error = False
        known_tokens = (Token.Text, Token.Literal.Number, Token.Name.Builtin,
                        Token.Punctuation, Token.Keyword)
        for tokenType, value in lexed:
            if tokenType in known_tokens:
                stripped = value.strip()
                if len(stripped) > 0:
                    words.append(value)
            else:
                print("AAAHHHH", tokenType, value)
                error = True
                raise ValueError("unrecognized token", tokenType, value)
                break
        # If the user starts the query with a question mark it forces running through model
        has_force_modeling_escape = False
        model_input_str = text
        if words[0].startswith("?"):
            words[0] = words[0][1:]
            has_force_modeling_escape = True
            model_input_str = model_input_str[1:]
        elif words[0] == "?":
            words = words[1:]
            has_force_modeling_escape = True
            model_input_str = model_input_str[1:]
        return ParseResult(was_error=error, words=words, model_input_str = model_input_str,
                           has_force_modeling_escape=has_force_modeling_escape)

@attr.s(frozen = True)
class ParseResult():
    words = attr.ib(converter=tuple)
    model_input_str = attr.ib()
    was_error = attr.ib(default=False)
    has_force_modeling_escape = attr.ib(default=False)
    def get_first_word(self):
        return self.words[0]
