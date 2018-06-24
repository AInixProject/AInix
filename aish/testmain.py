from xonsh import main as xmain
from xonsh.main import XonshMode
import builtins
import pudb
import testshell
from xonsh.platform import HAS_PYGMENTS, ON_WINDOWS
from xonsh.jobs import ignore_sigtstp
import os
import signal

def main(argv=None):
    args = None
    try:
        args = xmain.premain(argv)
        realshell = builtins.__xonsh_shell__.shell 
        builtins.__xonsh_shell__.shell = testshell.AishShell()
        return main_aish(args)
    except Exception as err:
        _failback_to_other_shells(args, err)

def main_aish(args):
    """Main entry point for aish cli. replaces main_xonsh in main.py of xonsh"""
    if not ON_WINDOWS:
        def func_sig_ttin_ttou(n, f):
            pass
        signal.signal(signal.SIGTTIN, func_sig_ttin_ttou)
        signal.signal(signal.SIGTTOU, func_sig_ttin_ttou)

    events.on_post_init.fire()
    env = builtins.__xonsh_env__
    shell = builtins.__xonsh_shell__
    try:
        if args.mode == XonshMode.interactive:
            # enter the shell
            env['XONSH_INTERACTIVE'] = True
            ignore_sigtstp()
            if (env['XONSH_INTERACTIVE'] and
                    not any(os.path.isfile(i) for i in env['XONSHRC'])):
                pass
                #print_welcome_screen()
            events.on_pre_cmdloop.fire()
            try:
                shell.shell.cmdloop()
            finally:
                events.on_post_cmdloop.fire()
        elif args.mode == XonshMode.single_command:
            # run a single command and exit
            run_code_with_cache(args.command.lstrip(), shell.execer, mode='single')
        elif args.mode == XonshMode.script_from_file:
            # run a script contained in a file
            path = os.path.abspath(os.path.expanduser(args.file))
            if os.path.isfile(path):
                sys.argv = [args.file] + args.args
                env['ARGS'] = sys.argv[:]  # $ARGS is not sys.argv
                env['XONSH_SOURCE'] = path
                shell.ctx.update({'__file__': args.file, '__name__': '__main__'})
                run_script_with_cache(args.file, shell.execer, glb=shell.ctx,
                                      loc=None, mode='exec')
            else:
                print('xonsh: {0}: No such file or directory.'.format(args.file))
        elif args.mode == XonshMode.script_from_stdin:
            # run a script given on stdin
            code = sys.stdin.read()
            run_code_with_cache(code, shell.execer, glb=shell.ctx, loc=None,
                                mode='exec')
    finally:
        events.on_exit.fire()
    postmain(args)

main(['--shell-type', 'readline'])
