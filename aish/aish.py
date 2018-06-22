from __future__ import unicode_literals
from prompt_toolkit import PromptSession
from pygments.lexers import BashLexer
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer

# Create prompt object.
session = PromptSession()
while True:
    text = session.prompt('$ ', lexer=PygmentsLexer(BashLexer))
    print(text)
