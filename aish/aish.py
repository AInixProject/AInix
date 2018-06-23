from __future__ import unicode_literals
from prompt_toolkit import PromptSession
from pygments.lexers import BashLexer
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer
import pygments
from pygments.token import Token
import builtin_cmds as bltin
import subprocess
import os


session = PromptSession()
myLexer = BashLexer()
env = { 'PWD': os.getcwd() }

    


def main():
    while True:
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

# cp from xonsh
def _change_working_directory(newdir, follow_symlinks=False):
    old = env['PWD']
    new = os.path.join(old, newdir)
    absnew = os.path.abspath(new)

    if follow_symlinks:
        absnew = os.path.realpath(absnew)

    try:
        os.chdir(absnew)
    except (OSError, FileNotFoundError):
        if new.endswith(get_sep()):
            new = new[:-1]
        if os.path.basename(new) == '..':
            env['PWD'] = new
    else:
        if old is not None:
            env['OLDPWD'] = old
        if new is not None:
            env['PWD'] = absnew

def cd(args, stdin=None, a=None, b=None, c=None):
    """Changes the directory.
    If no directory is specified (i.e. if `args` is None) then this
    changes to the current user's home directory.
    """
    oldpwd = env.get('OLDPWD', None)
    cwd = env['PWD']

    follow_symlinks = False
    if len(args) > 0 and args[0] == '-P':
        follow_symlinks = True
        del args[0]

    if len(args) == 0:
        d = os.path.expanduser('~')
    elif len(args) == 1:
        d = os.path.expanduser(args[0])
        if not os.path.isdir(d):
            if d == '-':
                if oldpwd is not None:
                    d = oldpwd
                else:
                    return '', 'cd: no previous directory stored\n', 1
            elif d.startswith('-'):
                try:
                    num = int(d[1:])
                except ValueError:
                    return '', 'cd: Invalid destination: {0}\n'.format(d), 1
                if num == 0:
                    return None, None, 0
                elif num < 0:
                    return '', 'cd: Invalid destination: {0}\n'.format(d), 1
                elif num > len(DIRSTACK):
                    e = 'cd: Too few elements in dirstack ({0} elements)\n'
                    return '', e.format(len(DIRSTACK)), 1
                else:
                    d = DIRSTACK[num - 1]
            else:
                d = _try_cdpath(d)
    else:
        return '', ('cd takes 0 or 1 arguments, not {0}. An additional `-P` '
                    'flag can be passed in first position to follow symlinks.'
                    '\n'.format(len(args))), 1
    if not os.path.exists(d):
        return '', 'cd: no such file or directory: {0}\n'.format(d), 1
    if not os.path.isdir(d):
        return '', 'cd: {0} is not a directory\n'.format(d), 1
    if not os.access(d, os.X_OK):
        return '', 'cd: permission denied: {0}\n'.format(d), 1

    # now, push the directory onto the dirstack if AUTO_PUSHD is set
    _change_working_directory(d, follow_symlinks)
    return None, None, 0

builtin_dict = { "echo": bltin.echo, "pwd": bltin.pwd, "cd": cd}

main()
