import xonsh
from xonsh.base_shell import BaseShell
from xonsh.readline_shell import ReadlineShell
import builtins

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer
import pygments
from pygments.token import Token
from pygments.lexers import BashLexer
import subprocess
import os

session = PromptSession()
myLexer = BashLexer()
builtin_dict = { }

class AishShell(BaseShell):
    def cmdloop(self, intro=None):
        while not builtins.__xonsh_exit__:
            env = builtins.__xonsh_env__
            text = session.prompt('$ ', lexer=PygmentsLexer(BashLexer))
            lexed = pygments.lex(text, myLexer)
            words = []
            error = False
            for tokenType, value in lexed:
                if tokenType in (Token.Text, Token.Literal.Number, Token.Name.Builtin):
                    stripped = value.strip()
                    if len(stripped) > 0:
                        words.append(value)
                else:
                    print("AAAHHHH", tokenType, value)
                    error = True
                    break
            if not error and len(words) > 0:
                if words[0] in builtin_dict:
                    builtin_dict[words[0]](words[1:], None, None, None, env)
                else:
                    try:
                        cmdProc = subprocess.Popen(words, cwd = env['PWD'])
                        cmdProc.wait()
                    except Exception as e:
                        print(e)
