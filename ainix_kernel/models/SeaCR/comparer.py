"""This module defines an interface for a comparer. In the SeaCR model, a
comparer is something that takes our current query and partially generated
AST and compares it to an example from the index, access whether or not
this example meets the current needs of the partially generated AST.

Most importantly this module defines an interface for comparers it use.
It also includes a couple comparers which are mostly for testing. For
an actually useful comparer see the torchcomparer module."""
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from ainix_kernel.models.SeaCR.treeutil import get_type_choice_nodes
from ainix_kernel.indexing.exampleindex import ExamplesIndex
from ainix_common.parsing.parseast import ObjectChoiceNode, AstNode, StringParser
import attr

from models.SeaCR.treeutil import get_type_choice_nodes
from parsing.parseast import ObjectChoiceNode, AstObjectChoiceSet


@attr.s(auto_attribs=True, frozen=True)
class ComparerResult:
    """The result from a Comparer comparison.

    Args:
        prob_valid_in_example : probability that the a valid implementation for
            the current leaf is present in the example
        impl_scores : A tuple listing tuples. Each Tuple represents an
            (score of how preferable this implementation is given a valid
            implementation exists in this example, implementation_name)
    """
    prob_valid_in_example: float
    impl_scores: Tuple[Tuple[float, str], ...]


class Comparer(ABC):
    """An ABC that defines the interface for a comparer."""
    @abstractmethod
    def compare(
        self,
        gen_query: str,
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
    ) -> ComparerResult:
        pass

    @abstractmethod
    def train(
        self,
        gen_query: str,
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
        expected_result: ComparerResult
    ):
        pass

    def get_parameters(self):
        return None


class SimpleRulebasedComparer(Comparer):
    """A deterministic comparer based off some simple heuristics"""
    def compare(
        self,
        gen_query: str,
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
    ) -> ComparerResult:
        potential_type_choice_nodes = get_type_choice_nodes(
            example_ast_root, gen_ast_current_leaf.get_type_to_choose_name())
        depth_diffs = self.get_impl_depth_difference(
            potential_type_choice_nodes, current_gen_depth)
        ranked_options = sorted(
            [(-score, name) for name, score in depth_diffs.items()],
            reverse=True
        )
        return ComparerResult(1, tuple(ranked_options))

    @staticmethod
    def get_impl_depth_difference(
        nodes_to_compare: List[Tuple[ObjectChoiceNode, int]],
        ref_depth: int
    ) -> Dict[str, float]:
        out = {}
        for node, depth_val in nodes_to_compare:
            node_impl_name = node.get_chosen_impl_name()
            depth_dif = abs(ref_depth - depth_val)
            out[node_impl_name] = min(out.get(node_impl_name, 9e9), depth_dif)
        return out

    def train(
        self,
        gen_query: str,
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
        expected_result: ComparerResult
    ):
        # RuleBasedComparer doesn't do any training
        pass


class OracleComparer(Comparer):
    """A comparer which peeks at the index to try and always returns the right
    results even if example not in training set. (Useful for testing)"""
    def __init__(self, index: ExamplesIndex):
        self.index = index
        self.parser = StringParser(self.index.type_context)

    def _get_actual_example_from_index(self, gen_query: str,
                                       gen_ast_current_leaf: ObjectChoiceNode):
        lookup_results = self.index.get_nearest_examples(
            gen_query, gen_ast_current_leaf.get_type_to_choose_name(), max_results=None)
        for result in lookup_results:
            if result.xquery == gen_query:
                return result
        raise ValueError(f"Oracle unable to find result for {gen_query}")

    def compare(
        self,
        gen_query: str,
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: ObjectChoiceNode,
    ) -> ComparerResult:
        oracle_gt_example = self._get_actual_example_from_index(gen_query, gen_ast_current_leaf)
        oracle_ast = self.parser.create_parse_tree(
            oracle_gt_example.ytext, oracle_gt_example.ytype)
        oracle_ast_set = AstObjectChoiceSet(oracle_ast.type_to_choose, None)
        oracle_ast_set.add(oracle_ast, True, 1, 1)
        return _create_gt_compare_result(
            example_ast_root, gen_ast_current_leaf, oracle_ast_set)


    def train(
        self,
        gen_query: str,
        gen_ast_current_root: ObjectChoiceNode,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
        expected_result: ComparerResult
    ):
        # Phhhsh, I'm an oracle. I don't need your "training"...
        pass


def _create_gt_compare_result(
    example_ast: ObjectChoiceNode,
    current_leaf: ObjectChoiceNode,
    ground_truth_set: AstObjectChoiceSet
) -> ComparerResult:
    """Creates a `CompareResult` based off some ground truth

    Args:
        example_ast : A parsed AST of the y_text inside the example we are creating
            the ground truth for.
        current_leaf : What we are currently generating for. Used to determine
            what kind of choice we are making.
        ground_truth_set : Our ground truth which we are making the result based
            off of.
    """
    choices_in_this_example = get_type_choice_nodes(
        example_ast, current_leaf.type_to_choose.name)
    right_choices = [e for e, depth in choices_in_this_example
                     if ground_truth_set.is_known_choice(e.get_chosen_impl_name())]
    in_this_example_impl_name_set = {c.get_chosen_impl_name()
                                     for c, depth in choices_in_this_example}
    right_choices_impl_name_set = {c.get_chosen_impl_name() for c in right_choices}
    this_example_right_prob = 1 if len(right_choices) > 0 else 0
    if right_choices:
        expected_impl_scores = [(1 if impl_name in right_choices_impl_name_set else 0, impl_name)
                                for impl_name in in_this_example_impl_name_set]
        expected_impl_scores.sort()
        expected_impl_scores = tuple(expected_impl_scores)
    else:
        expected_impl_scores = None
    return ComparerResult(this_example_right_prob, expected_impl_scores)