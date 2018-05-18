"""Module defines several useful for evaluating effectiveness of the system"""
from __future__ import print_function
from ignite.metrics import CategoricalAccuracy, Loss, Metric
import pudb
from cmd_parse import ProgramNode, EndOfCommandNode
from colorama import init
init()
from colorama import Fore, Back, Style
class BashMetric(Metric):
    def reset(self):
        self._num_examples = 0
        # a command may contain multiple programs if compound
        self._num_programs_seen = 0
        self._num_first_program_right = 0
        self._num_programs_right = 0
        self._num_args_seen = 0
        self._num_args_seen_when_right = 0
        self._num_args_pres_correct = 0
        self._num_exact_match = 0
        self._num_values_seen = 0
        self._num_values_exact_match = 0

        self.seen_query_set = set()

    def update(self, output):
        y_pred, y, queries = output
        self._num_examples += len(y_pred)
        for p, gt, query in zip(y_pred, y, queries):
            isFirstCommand = True
            fullMatch = True
            for i, gtCmd in enumerate(gt):
                pCmd = p[i] if i < len(p) else EndOfCommandNode()
                gotNodeRight = type(pCmd) == type(gtCmd)
                if not gotNodeRight:
                    fullMatch = False
                
                if isinstance(gtCmd, ProgramNode):
                    self._num_programs_seen += 1
                    gotCommand = gotNodeRight and pCmd.program_desc.name == gtCmd.program_desc.name
                    self._num_args_seen += len(gtCmd.arguments)
                    if gotCommand:
                        if isFirstCommand:
                            self._num_first_program_right += 1
                        self._num_programs_right += 1
                        self._num_args_seen_when_right += len(gtCmd.arguments)
                        for pArg, gtArg in zip(pCmd.arguments, gtCmd.arguments):
                            if pArg.present == gtArg.present:
                                self._num_args_pres_correct += 1
                            else:
                                fullMatch = False

                            if gtArg.value is not None:
                                self._num_values_seen += 1
                                #print("expected value", gtArg.value, " got arg value", pArg.value)
                                if gtArg.value == pArg.value:
                                    self._num_values_exact_match += 1
                                else:
                                    fullMatch = False
                    else:
                        fullMatch = False
                isFirstCommand = False

            if fullMatch:
                self._num_exact_match += 1

            # Print some log information on whether we got it right or wrong
            # The seen_query_set is used because we duplicate entries
            # as a hacky way of doing the replacments. Examples without
            # any replacements might appear multiple times. We don't want to
            # print them again
            if query not in self.seen_query_set:
                print("---")
                print(query)
                if fullMatch:
                    print(Fore.GREEN, end='')
                    if gt.as_shell_string() != p.as_shell_string():
                        print(" Expected:", gt.as_shell_string())
                    print("Predicted:", p.as_shell_string())
                else:
                    print(Fore.RED, end='')
                    print(" Expected:", gt.as_shell_string())
                    print("Predicted:", p.as_shell_string())
                print(Style.RESET_ALL, end='')
                self.seen_query_set.add(query)


    def first_cmd_acc(self):
        return self._num_first_program_right / self._num_examples

    def arg_acc(self):
        if self._num_args_seen > 0:
            return self._num_args_pres_correct / self._num_args_seen
        else:
            return None

    def arg_val_exact_acc(self):
        if self._num_values_seen == 0:
            return float('nan')
        return self._num_values_exact_match / self._num_values_seen

    def exact_match_acc(self):
        return self._num_exact_match / self._num_examples

    def compute(self):
        pass

