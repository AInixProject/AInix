"""Implements a simple echo command for xonsh."""


def echo(args, stdin, stdout, stderr, env):
    """A simple echo command."""
    opts = _echo_parse_args(args)
    if opts is None:
        return
    if opts['help']:
        print(ECHO_HELP, file=stdout)
        return 0
    ender = opts['end']
    args = map(str, args)
    if opts['escapes']:
        args = map(lambda x: x.encode().decode('unicode_escape'), args)
    print(*args, end=ender, file=stdout)
    return 0, None


def _echo_parse_args(args):
    out = {'escapes': False, 'end': '\n', 'help': False}
    if '-e' in args:
        args.remove('-e')
        out['escapes'] = True
    if '-E' in args:
        args.remove('-E')
        out['escapes'] = False
    if '-n' in args:
        args.remove('-n')
        out['end'] = ''
    if '-h' in args or '--help' in args:
        out['help'] = True
    return out


ECHO_HELP = """Usage: echo [OPTIONS]... [STRING]...
Echo the STRING(s) to standard output.
  -n             do not include the trailing newline
  -e             enable interpretation of backslash escapes
  -E             disable interpretation of backslash escapes (default)
  -h  --help     display this message and exit
This version of echo was written in Python for the xonsh project: http://xon.sh
Based on echo from GNU coreutils: http://www.gnu.org/software/coreutils/"""

"""A pwd implementation for xonsh."""
import os


def pwd(args, stdin, stdout, stderr, env):
    """A pwd implementation"""
    e = env['PWD']
    if '-h' in args or '--help' in args:
        print(PWD_HELP, file=stdout)
        return 0
    if '-P' in args:
        e = os.path.realpath(e)
    print(e, file=stdout)
    return 0, None


PWD_HELP = """Usage: pwd [OPTION]...
Print the full filename of the current working directory.
  -P, --physical   avoid all symlinks
      --help       display this help and exit
This version of pwd was written in Python for the xonsh project: http://xon.sh
Based on pwd from GNU coreutils: http://www.gnu.org/software/coreutils/"""


# Not Implemented
#   -L, --logical    use PWD from environment, even if it contains symlinks

