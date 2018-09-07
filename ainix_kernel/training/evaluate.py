from abc import ABC, abstractmethod
from typing import Dict
from parseast import ObjectChoiceNode


class Evaluation(ABC):
    @abstractmethod
    def get_data(self) -> Dict:
        pass


class AstEvaluation(Evaluation):
    def __init__(self, prediction: ObjectChoiceNode, ground_truth: ObjectChoiceNode):
        self.data = {}
        self.prediction = prediction
        self.ground_truth = ground_truth
        self._do_eval()

    def _do_eval(self):
        if self.prediction != self.ground_truth and self.prediction is not None and \
                self.prediction.dump_str() == self.ground_truth.dump_str():
            print(self.prediction.dump_str())
            print(self.ground_truth.dump_str())
            print("break")

        self.data["ExactMatch"] = self.prediction == self.ground_truth

    def get_data(self) -> Dict:
        return self.data


class BinaryStat:
    def __init__(self, name: str, first_true: bool = None):
        self.name = name
        self.total_count: int = 0
        self.true_count: int = 0
        if first_true is not None:
            self.total_count += 1
            if first_true:
                self.true_count += 1

    @property
    def percent_true_str(self) -> str:
        if self.total_count == 0:
            return "N/A"
        return f"{(self.true_count / self.total_count)*100:.2f}%"

    def __add__(self, other):
        self.total_count += other.total_count
        self.true_count += other.true_count
        return self


def value_to_stat(name, value):
    if isinstance(value, bool):
        return BinaryStat(name, value)
    raise ValueError(f"Unrecognized value {value} of name {name}")


class EvaluateLogger:
    def __init__(self):
        self.stats = {}

    def add_evaluation(self, evaluation: Evaluation, epoch: int = 0):
        for name, value in evaluation.get_data().items():
            as_stat = value_to_stat(name, value)
            if name not in self.stats:
                self.stats[name] = as_stat
            else:
                self.stats[name] += as_stat


def print_ast_eval_log(eval_logger: EvaluateLogger):
    stats = eval_logger.stats
    exact_match_percent = stats['ExactMatch'].percent_true_str
    print(f"Exact Match Percent {exact_match_percent}")

