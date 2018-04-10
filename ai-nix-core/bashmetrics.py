"""Module defines several useful for evaluating effectiveness of the system"""
from ignite.metrics import CategoricalAccuracy, Loss, Metric
class BashMetric(Metric):
    def reset(self):
        self._num_examples = 0
        self._num_first_commands_right = 0
        self._num_args_seen = 0
        self._num_args_seen_when_right = 0
        self._num_args_pres_correct = 0
        self._num_exact_match = 0
        self._num_values_seen = 0
        self._num_values_exact_match = 0

    def update(self, output):
        y_pred, y = output
        self._num_examples += len(y_pred)
        for p, gt in zip(y_pred, y):
            pFirstCmd, gtFirstCmd = p[0], gt[0]
            gotFirstCommand = pFirstCmd.program_desc.name == gtFirstCmd.program_desc.name
            self._num_args_seen += len(gtFirstCmd.arguments)
            if gotFirstCommand:
                self._num_first_commands_right += 1
                self._num_args_seen_when_right += len(gtFirstCmd.arguments)
                fullMatch = True
                for pArg, gtArg in zip(pFirstCmd.arguments, gtFirstCmd.arguments):
                    if pArg.present == gtArg.present:
                        self._num_args_pres_correct += 1
                    else:
                        fullMatch = False

                    if gtArg.value is not None:
                        self._num_values_seen += 1
                        print("expected value", gtArg.value, " got arg value", pArg.value)
                        if gtArg.value == pArg.value:
                            self._num_values_exact_match += 1
                        else:
                            fullMatch = False

                if fullMatch:
                    self._num_exact_match += 1


    def first_cmd_acc(self):
        return self._num_first_commands_right / self._num_examples

    def arg_acc(self):
        if self._num_args_seen > 0:
            return self._num_args_pres_correct / self._num_args_seen
        else:
            return None

    def exact_match_acc(self):
        return self._num_exact_match / self._num_examples

    def compute(self):
        pass

