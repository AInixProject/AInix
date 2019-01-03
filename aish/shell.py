print("Preparing shell...")
import xonsh
from xonsh.base_shell import BaseShell
from xonsh.ptk2.shell import PromptToolkit2Shell
import builtins

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer
import pygments
from pygments.lexers import BashLexer
import subprocess
import os
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from ainix_kernel.interfacing.shellinterface import Interface
from aish.execution_classifier import ExecutionClassifier
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import aish.execer
from aish.parser import BashParser

builtin_dict = {}


class AishShell(BaseShell):
    def __init__(self):
        self.session = PromptSession()
        self.parser = BashParser()
        self.kernel_interface = Interface("../ainix_kernel/training/saved_model.pt")
        self.exec_classifier = ExecutionClassifier()
        self.exec_function = aish.execer.execute

    def singleline(self, store_in_history=True, auto_suggest=None,
                   enable_history_search=True, multiline=True, **kwargs):
        """Reads a single line of input from the shell. The store_in_history
        kwarg flags whether the input should be stored in PTK's in-memory
        history.
        """
        events.on_pre_prompt.fire()
        env = builtins.__xonsh_env__
        line = self.session.prompt('$ ', lexer=PygmentsLexer(BashLexer))
        events.on_post_prompt.fire()
        return line

    def cmdloop(self, intro=None):
        while not builtins.__xonsh_exit__:
            env = builtins.__xonsh_env__
            env['XONSH_SHOW_TRACEBACK'] = True
            text = self.singleline()
            parse = self.parser.parse(text)
            if parse.was_error:
                continue

            exec_type = self.exec_classifier.classify_string(parse)
            if False and words[0] in builtin_dict:
                builtin_dict[words[0]](words[1:], None, None, None, env)
            else:
                try:
                    if exec_type.run_through_model:
                        prediction = self.kernel_interface.predict(text)
                        print("predict:", prediction)
                    else:
                        self.exec_function(parse)
                except Exception as e:
                    print(e)


class AishShell2(PromptToolkit2Shell):
    def __init__(self, **kwargs):
        super().__init__(execer=None, ctx=None, **kwargs)

        self.parser = BashParser()
        self.kernel_interface = Interface("../ainix_kernel/training/saved_model.pt")
        self.exec_classifier = ExecutionClassifier()
        self.exec_function = aish.execer.execute

    def cmdloop(self, intro=None):
        """Enters a loop that reads and execute input from user."""
        if intro:
            print(intro)
        auto_suggest = AutoSuggestFromHistory()
        while not builtins.__xonsh__.exit:
            try:
                line = self.singleline(auto_suggest=auto_suggest)
                parse = self.parser.parse(line)
                if parse.was_error:
                    continue
                exec_type = self.exec_classifier.classify_string(parse)
                try:
                    if exec_type.run_through_model:
                        print("running on", parse.model_input_str)
                        prediction = self.kernel_interface.predict(parse.model_input_str, "Program")
                        print("predict:", prediction)
                    else:
                        self.exec_function(parse)
                except Exception as e:
                    print(e)
                    raise e
                #if not line:
                #    self.emptyline()
                #else:
                #    line = self.precmd(line)
                #    self.default(line)
            except (KeyboardInterrupt, SystemExit):
                self.reset_buffer()
            except EOFError:
                if builtins.__xonsh_env__.get("IGNOREEOF"):
                    print('Use "exit" to leave the shell.', file=sys.stderr)
                else:
                    break

