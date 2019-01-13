from abc import ABC, abstractmethod
from typing import Dict, List, Set
from ainix_common.parsing.ast_components import ObjectChoiceNode, AstObjectChoiceSet
from ainix_common.parsing.model_specific.tokenizers import StringTokenizer
from ainix_common.parsing.stringparser import AstUnparser
from colorama import Fore, Back, Style
import warnings


class Evaluation(ABC):
    @abstractmethod
    def get_data(self) -> Dict:
        pass


class AstEvaluation(Evaluation):
    def __init__(self, prediction: ObjectChoiceNode, ground_truth: AstObjectChoiceSet,
                 y_texts: Set[str], x_text: str, exception, unparser: AstUnparser):
        self.data = {}
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.y_texts = y_texts
        self.x_text = x_text
        self.p_exception = exception
        if self.prediction is not None:
            self.predicted_y = unparser.to_string(self.prediction, self.x_text).total_string
        else:
            self.predicted_y = f"EXCEPTION {str(self.p_exception)}"
        self.in_ast_set = self.ground_truth.is_node_known_valid(self.prediction)
        self.correct = self.in_ast_set or self.predicted_y in self.y_texts
        if self.correct and not self.in_ast_set:
            warnings.warn(f"The prediction is not in ground truth but value "
                          f"matches a y string. "
                          f"Prediction text {self.predicted_y} actuals {self.y_texts}")
        self._fill_stats()

    def print_vals(self, unparser: AstUnparser):
        y_texts_v = list(self.y_texts)[0] if len(self.y_texts) == 1 else self.y_texts
        print("---")
        print(self.x_text)
        if self.correct:
            if self.correct and not self.in_ast_set:
                print(Fore.YELLOW, end='')
            else:
                print(Fore.GREEN, end='')
            if self.predicted_y not in self.y_texts:
                print(" Expected:", y_texts_v)
        else:
            print(Fore.RED, end='')
            print(" Expected:", y_texts_v)
        print("Predicted:", self.predicted_y)
        print(Style.RESET_ALL, end='')

    def _fill_stats(self):
        self.data["ExactMatch"] = self.correct

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
        return f"{self.name}: {self.percent_true_str}"

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
            #self.all_evals.append(evaluation)


def print_ast_eval_log(eval_logger: EvaluateLogger):
    stats = eval_logger.stats
    exact_match_percent = stats['ExactMatch'].percent_true_str
    print(f"Exact Match Percent {exact_match_percent}")
    #if print_all_evals:
    #    for eval in eval_logger.all_evals:
    #        print(f"x: {eval.x_text}")
    #        print(f"y: {eval.x_text}")
