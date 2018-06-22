from __future__ import unicode_literals
from prompt_toolkit import PromptSession
from pygments.lexers import BashLexer
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer
import pygments
from pygments.token import Token
import subprocess


session = PromptSession()
myLexer = BashLexer()

while True:
    text = session.prompt('$ ', lexer=PygmentsLexer(BashLexer))
    lexed = pygments.lex(text, myLexer)
    words = []
    error = False
    for tokenType, value in lexed:
        if tokenType in (Token.Text, Token.Literal.Number):
            stripped = value.strip()
            if len(stripped) > 0:
                words.append(value)
        else:
            print("AAAHHHH", tokenType, value)
            error = True
            break
    if not error:
        try:
            cmdProc = subprocess.Popen(words)
            cmdProc.wait()
        except Exception as e:
            print(e)
