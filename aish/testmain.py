from xonsh import main as xmain
import builtins
import pudb
import testshell

class InjectShell(object):
    """Taken from shell.py in xonsh. 
    Main xonsh shell.

    Initializes execution environment and decides if prompt_toolkit or
    readline version of shell should be used.
    """

    shell_type_aliases = {
        'b': 'best',
        'ptk': 'prompt_toolkit',
        'ptk1': 'prompt_toolkit1',
        'ptk2': 'prompt_toolkit2',
        'prompt-toolkit': 'prompt_toolkit',
        'prompt-toolkit1': 'prompt_toolkit1',
        'prompt-toolkit2': 'prompt_toolkit2',
        'rand': 'random',
        'rl': 'readline',
        }

    def __init__(self, execer, ctx=None, shell_type=None, **kwargs):
        """
        Parameters
        ----------
        execer : Execer
            An execer instance capable of running xonsh code.
        ctx : Mapping, optional
            The execution context for the shell (e.g. the globals namespace).
            If none, this is computed by loading the rc files. If not None,
            this no additional context is computed and this is used
            directly.
        shell_type : str, optional
            The shell type to start, such as 'readline', 'prompt_toolkit1',
            or 'random'.
        """
        print("Sneaky sneak")
        self.execer = execer
        self.ctx = {} if ctx is None else ctx
        env = builtins.__xonsh_env__
        # build history backend before creating shell
        builtins.__xonsh_history__ = hist = xhm.construct_history(
            env=env.detype(), ts=[time.time(), None], locked=True)

        # pick a valid shell -- if no shell is specified by the user,
        # shell type is pulled from env
        if shell_type is None:
            shell_type = env.get('SHELL_TYPE')
            if shell_type == 'none':
                # This bricks interactive xonsh
                # Can happen from the use of .xinitrc, .xsession, etc
                shell_type = 'best'
        shell_type = self.shell_type_aliases.get(shell_type, shell_type)
        if shell_type == 'best' or shell_type is None:
            shell_type = best_shell_type()
        elif shell_type == 'random':
            shell_type = random.choice(('readline', 'prompt_toolkit'))
        if shell_type == 'prompt_toolkit':
            if not has_prompt_toolkit():
                warnings.warn('prompt_toolkit is not available, using '
                              'readline instead.')
                shell_type = 'readline'
            elif not ptk_version_is_supported():
                warnings.warn('prompt-toolkit version < v1.0.0 is not '
                              'supported. Please update prompt-toolkit. Using '
                              'readline instead.')
                shell_type = 'readline'
            else:
                shell_type = ptk_shell_type()
        self.shell_type = env['SHELL_TYPE'] = shell_type
        # actually make the shell
        if shell_type == 'none':
            from xonsh.base_shell import BaseShell as shell_class
        elif shell_type == 'prompt_toolkit2':
            from xonsh.ptk2.shell import PromptToolkit2Shell as shell_class
        elif shell_type == 'prompt_toolkit1':
            from xonsh.ptk.shell import PromptToolkitShell as shell_class
        elif shell_type == 'readline':
            from xonsh.readline_shell import ReadlineShell as shell_class
        else:
            raise XonshError('{} is not recognized as a shell type'.format(
                             shell_type))
        self.shell = shell_class(execer=self.execer,
                                 ctx=self.ctx, **kwargs)
        # allows history garbage collector to start running
        if hist.gc is not None:
            hist.gc.wait_for_shell = False

#xmain.Shell = InjectShell

def main(argv=None):
    args = None
    try:
        args = xmain.premain(argv)
        realshell = builtins.__xonsh_shell__.shell 
        builtins.__xonsh_shell__.shell = testshell.AishShell(realshell.execer, realshell.ctx)
        return xmain.main_xonsh(args)
    except Exception as err:
        _failback_to_other_shells(args, err)

main(['--shell-type', 'readline'])
