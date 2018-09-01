from abc import ABC, abstractmethod
from models.model_types import StringTypeTranslateCF
from indexing.exampleindex import ExamplesIndex, Example
from ainix_common.parsing.parseast import AstNode, ObjectNode, \
    ObjectChoiceNode, ArgPresentChoiceNode, StringParser, ObjectChoiceLikeNode
from ainix_common.parsing.typecontext import AInixType, AInixObject
from typing import List, Tuple, Union, Dict
import attr


class SeaCRModel(StringTypeTranslateCF):
    def __init__(
        self,
        index: ExamplesIndex,
        type_predictor: 'TypePredictor' = None,
    ):
        self.type_context = index.type_context
        comparer = SimpleRulebasedComparer()
        self.type_predictor = type_predictor if type_predictor else \
            TypePredictor(index, comparer)

    def predict(self, x_string: str, y_type_name: str):
        root_type = self.type_context.get_type_by_name(y_type_name)
        root_node = ObjectChoiceNode(root_type, None)
        self.predict_step(x_string, root_node, root_type)
        return root_node

    def predict_step(self, x_query: str, current_leaf: AstNode, root_y_type: AInixType):
        if isinstance(current_leaf, ObjectChoiceLikeNode):
            next_nodes = [self.type_predictor.predict(x_query, current_leaf)]
        elif isinstance(current_leaf, ObjectNode):
            next_nodes = current_leaf.get_children()
        else:
            raise ValueError(f"leaf node {current_leaf} not predictable")
        next_nodes = [node for node in next_nodes if node is not None]

        for next_node in next_nodes:
            self.predict_step(x_query, next_node, root_y_type)


def _get_impl_names_from_indexed_rep(
        index_rep: str,
        type_to_choose: str
) -> List[str]:
    # TODO (DNGros): This is disgusting and shouldnt exist
    out = []
    for s in index_rep.split(f"CLASSIFY_TYPE={type_to_choose}")[1:]:
        o_rep = s.split()[1]
        o_name = o_rep.split("=")[1]
        out.append(o_name)
    return out


class TypePredictor:
    def __init__(self, index: ExamplesIndex, comparer: 'Comparer'):
        self.index = index
        self.type_context = index.type_context
        self.comparer = comparer

    def compare_example(
        self,
        x_query: str,
        current_leaf: ObjectChoiceLikeNode,
        example: Example
    ) -> 'ComparerResult':
        # TODO (DNGros): cache the parser and memoize the parsing during training
        parser = StringParser(self.type_context.get_type_by_name(example.ytype))
        example_ast = parser.create_parse_tree(example.ytext)
        return self.comparer.compare(x_query, current_leaf, example.xquery, example_ast)


    def predict(
        self,
        x_query: str,
        current_leaf: ObjectChoiceLikeNode
    ) -> AstNode:
        if current_leaf.get_depth() > 20:
            raise ValueError("whoah, that's too deep man")
        type_name = current_leaf.get_type_to_choose_name()
        search_results = self.index.get_nearest_examples(x_query, choose_type_name=type_name)
        comparer_result = self.compare_example(x_query, current_leaf, search_results[0])
        choose_name = comparer_result.impl_scores[0][0]
        return current_leaf.gen_set_valid_choice_by_name(choose_name)


@attr.s(auto_attribs=True, frozen=True)
class ComparerResult:
    """The result from a Comparer comparison.

    Attributes:
        prob_valid_in_example : probability that the a valid implementation for
            the current leaf is present in the example
        impl_scores : A tuple listing tuples. Each Tuple represents an
            (implementation name, score of how preferable this implementation is
             given a valid implementation exists in this example)
    """
    prob_valid_in_example: float
    impl_scores: Tuple[Tuple[str, float], ...]


class Comparer(ABC):
    @abstractmethod
    def compare(
        self,
        gen_query: str,
        gen_ast_current_leaf: ObjectChoiceLikeNode,
        example_query: str,
        example_ast_root: ObjectChoiceNode,
    ) -> ComparerResult:
        pass


def _get_type_choice_nodes(root_node: AstNode, type_name: str) -> List[ObjectChoiceLikeNode]:
    out = []

    def check_type(cur_node: AstNode):
        if cur_node is None:
            return
        if isinstance(cur_node, ObjectChoiceLikeNode):
            if cur_node.get_type_to_choose_name() == type_name:
                out.append(cur_node)
        for child in cur_node.get_children():
            check_type(child)
    check_type(root_node)
    return out


class SimpleRulebasedComparer(Comparer):
    def compare(
        self,
        gen_query: str,
        gen_ast_current_leaf: ObjectChoiceLikeNode,
        example_query: str,
        example_ast_root: ObjectChoiceNode,
    ) -> ComparerResult:
        potential_type_choice_nodes = _get_type_choice_nodes(
            example_ast_root, gen_ast_current_leaf.get_type_to_choose_name())
        depth_diffs = self.get_impl_depth_difference(
            potential_type_choice_nodes, gen_ast_current_leaf)
        inverse_depth_diffs = {name: -d for name, d in depth_diffs.items()}
        ranked_options = sorted(inverse_depth_diffs.items(), key=lambda t: t[1], reverse=True)
        return ComparerResult(1, tuple(ranked_options))

    def get_impl_depth_difference(
        self,
        nodes_to_compare: List[ObjectChoiceLikeNode],
        reference_node: ObjectChoiceLikeNode
    ) -> Dict[str, float]:
        ref_depth = reference_node.get_depth()
        depth_differences = [abs(node.get_depth() - ref_depth) for node in nodes_to_compare]
        out = {}
        for node, depth_val in zip(nodes_to_compare, depth_differences):
            node_impl_name = node.get_chosen_impl_name()
            out[node_impl_name] = min(out.get(node_impl_name, 9e9), depth_val)
        return out

