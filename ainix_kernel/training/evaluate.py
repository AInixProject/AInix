from abc import ABC, abstractmethod
from typing import Dict
from ainix_common.parsing.parseast import ObjectChoiceNode, AstObjectChoiceSet


class Evaluation(ABC):
    @abstractmethod
    def get_data(self) -> Dict:
        pass


class AstEvaluation(Evaluation):
    def __init__(self, prediction: ObjectChoiceNode, ground_truth: AstObjectChoiceSet):
        self.data = {}
        self.prediction = prediction
        self.ground_truth = ground_truth
        self._do_eval()

    def _do_eval(self):
        self.data["ExactMatch"] = self.ground_truth.is_node_known_valid(self.prediction)

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
    def true_frac(self) -> float:
        """Fraction true. Expressed in interval [0, 1]"""
        if self.total_count == 0:
            return None
        return self.true_count / self.total_count

    @property
    def percent_true_str(self) -> str:
        if self.total_count == 0:
            return "N/A"
        return f"{(self.true_count / self.total_count)*100:.2f}%"

    def __str__(self):
        return f"{name}: {self.percent_true_str}"

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
