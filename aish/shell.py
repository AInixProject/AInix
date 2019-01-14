import sys, os

from strformat_utils import get_highlighted_text

print("Preparing shell...")
sys.path.insert(0, os.path.abspath('..'))
from typing import Tuple, Sequence
from xonsh.ptk2.shell import PromptToolkit2Shell
import builtins
from ainix_kernel.interfacing.shellinterface import Interface
from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.stringparser import UnparseResult
from ainix_kernel.models.model_types import ExampleRetrieveExplanation
from ainix_kernel.explan_tools.example_explan import post_process_explanations
from aish.execution_classifier import ExecutionClassifier
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import aish.execer
from aish.parser import BashParser
from terminaltables import SingleTable
import colorama

builtin_dict = {}

class AishShell2(PromptToolkit2Shell):
    def __init__(self, **kwargs):
        super().__init__(execer=None, ctx=None, **kwargs)

        self.parser = BashParser()
        print("Loading model...")
        print("(NOTE, this currently super inefficient and takes quite a while.\n"
              " It could be optimized a lot).")
        self.kernel_interface = Interface("../ainix_kernel/training/saved_model.pt")
        self.exec_classifier = ExecutionClassifier()
        self.exec_function = aish.execer.execute
        print("model loaded.")

    def do_example_retrieve_explanation(
        self,
        retr_explans: Tuple[ExampleRetrieveExplanation, ...],
        outputted_ast: ObjectChoiceNode,
        outputted_unparse: UnparseResult
    ):
        print("How each part of the output was derived:")
        post_procs = post_process_explanations(
            retr_explans,
            # Hackily just grab things out of the interface. This should be improved
            self.kernel_interface.example_store,
            outputted_ast,
            outputted_unparse
        )
        print(post_procs)
        headers = ("Parts of Output", "Reference Y", "Reference X")
        rows = [
            (
                get_highlighted_text(
                    string=outputted_unparse.total_string,
                    include_intervals=[(interval.begin, interval.end)
                                       for interval in p.input_str_intervals]
                ),
                p.example_cmd,
                p.example_str
            )
            for p in post_procs
        ]
        table = SingleTable([headers] + rows)
        print(table.table)

    def do_predict(self, in_x: str):
        #print(f"model: {in_x}")
        pred_result = self.kernel_interface.predict(
            in_x, "CommandSequence")
        if pred_result.success:
            print(f"predict: {pred_result.unparse.total_string} "
                  f"(confidence score {pred_result.metad.total_confidence*10:.1f})")
            self.do_example_retrieve_explanation(
                pred_result.metad.example_retrieve_explanations, pred_result.ast,
                pred_result.unparse)
        else:
            print("Model encountered an error while predicting:")
            print(f"{pred_result.error_message}")

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
                        self.do_predict(parse.model_input_str)
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



