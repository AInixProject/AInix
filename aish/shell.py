import sys, os

from man_explain import print_man_page_explan
from strformat_utils import get_highlighted_text, get_only_text_in_intervals

print("Preparing shell...")
sys.path.insert(0, os.path.abspath('..'))
from typing import Tuple, Sequence, Optional
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
import math

builtin_dict = {}

PROMPT_CONF_THRESHOLD = 0.8


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
        if retr_explans is None or len(retr_explans) == 0:
            return
        print("How each part of the output was derived:")
        post_procs = post_process_explanations(
            retr_explans,
            # Hackily just grab things out of the interface. This should be improved
            self.kernel_interface.example_store,
            outputted_ast,
            outputted_unparse
        )
        headers = ("Parts of Output", "Reference Y", "Reference X")
        rows = [
            (
                get_only_text_in_intervals(
                    string=outputted_unparse.total_string,
                    include_intervals=p.input_str_intervals
                ),
                p.example_cmd,
                p.example_str
            )
            for p in post_procs
        ]
        table = SingleTable([headers] + rows)
        print(table.table)

    def do_predict(self, in_x: str) -> Tuple[Optional[str], float]:
        #print(f"model: {in_x}")
        pred_result = self.kernel_interface.predict(
            in_x, "CommandSequence")
        if pred_result.success:
            conf_emoji = ""
            total_confidence = pred_result.metad.total_confidence
            if total_confidence < 0.2:
                conf_emoji = "❌"
            elif total_confidence < 0.5:
                conf_emoji = "⚠️"
            elif total_confidence > 0.925:
                conf_emoji = "✅"
            print(f"predict: "
                  f"{colorama.Fore.MAGENTA}{pred_result.unparse.total_string.strip()}"
                  f"{colorama.Fore.RESET} "
                  f"(confidence score {pred_result.metad.total_confidence:.2f} {conf_emoji} )")
            # actual method
            #self.do_example_retrieve_explanation(
            #    pred_result.metad.example_retrieve_explanations, pred_result.ast,
            #    pred_result.unparse)

            # hacky for fullrett

            # Hackily just grab things out of the interface. This should be improved
            #index = self.kernel_interface.example_store
            #retr_explan = pred_result.metad.example_retrieve_explanations[0]
            #for sim, example_id in zip(
            #    retr_explan.reference_confidence,
            #    retr_explan.reference_example_ids
            #):
            #    example = index.get_example_by_id(example_id)
            #    yvs = index.get_y_values_for_y_set(example.y_set_id)
            #    print(math.exp(sim), yvs[0].y_text)
            print("Other possible options:")
            printed = set([pred_result.unparse.total_string.strip()])
            for sim, other_ast in pred_result.metad.other_options[1:]:
                try:
                    s = self.kernel_interface.unparser.to_string(other_ast).total_string.strip()
                    if s in printed:
                        continue
                    print(math.exp(sim), s)
                    printed.add(s)
                except Exception:
                    # panic! time
                    pass
                if len(printed) >= 3:
                    break
            print("-")

            if total_confidence > PROMPT_CONF_THRESHOLD:
                pass
            return pred_result.unparse.total_string.strip(), total_confidence
        else:
            print("Model encountered an error while predicting:")
            print(f"{pred_result.error_message}")
            return None, 0

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
                        pred_str, confidence = self.do_predict(parse.model_input_str)
                        if confidence > PROMPT_CONF_THRESHOLD:
                            print()
                            print_man_page_explan(pred_str, parse.model_input_str)
                            if self.ask_user_confirm_exec(pred_str):
                                print("")
                                self.exec_function(self.parser.parse(pred_str))
                        else:
                            print("Model output did not meet confidence threshold "
                                  "prompt whether to execute.")
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

    def ask_user_confirm_exec(self, str_to_exec) -> bool:
        print(f'Would you like to execute "{str_to_exec}"? "yes" to confirm: ')
        user_input = input()
        return user_input == "yes"


