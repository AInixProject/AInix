from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from ainix_kernel.models.SeaCR.treeutil import get_type_choice_nodes
from ainix_common.parsing.parseast import ObjectChoiceNode, AstNode
import attr


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
