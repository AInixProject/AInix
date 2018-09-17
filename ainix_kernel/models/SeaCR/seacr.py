from abc import ABC, abstractmethod
from models.model_types import StringTypeTranslateCF, ModelCantPredictException
from indexing.exampleindex import ExamplesIndex
from indexing.examplestore import Example, DataSplits
from ainix_common.parsing.parseast import AstNode, ObjectNode, \
    ObjectChoiceNode, StringParser,  \
    AstObjectChoiceSet, AstSet, ObjectNodeSet
from ainix_common.parsing.typecontext import AInixType, AInixObject
from typing import List, Tuple, Union, Dict, Optional
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

    def predict(
        self,
        x_string: str,
        y_type_name: str,
        use_only_train_data: bool
    ) -> AstNode:  # TODO (DNGros): change to set
        root_type = self.type_context.get_type_by_name(y_type_name)
        root_node = ObjectChoiceNode(root_type)
        self._predict_step(x_string, root_node, root_type, use_only_train_data, 0)
        root_node.freeze()
        return root_node

    def _predict_step(
        self,
        x_query: str,
        current_leaf: AstNode,
        root_y_type: AInixType,
        use_only_train_data: bool,
        current_depth: int
    ):
        if isinstance(current_leaf, ObjectChoiceNode):
            predicted_impl = self.type_predictor.predict(
                x_query, current_leaf, use_only_train_data, current_depth)
            new_node = ObjectNode(predicted_impl)
            current_leaf.set_choice(new_node)
            if new_node is not None:
                self._predict_step(x_query, new_node, root_y_type,
                                   use_only_train_data, current_depth + 1)
        elif isinstance(current_leaf, ObjectNode):
            # TODO (DNGros): this is messy. Should have better iteration based
            # off next unfilled node rather than having to mutate state.
            #####
            # Actually it would probably be better to just store a pointer to
            # the root and a pointer to the leaf. Then you call an add_to_leaf()
            # on the root which does a copying add, with structure sharing of
            # any arg before the arg that leaf is on. This is also nice because
            # then AstNodes can be made purly immutable and beam search becomes
            # nearly trivial to do
            for arg in current_leaf.implementation.children:
                if not arg.required:
                    new_node = ObjectChoiceNode(arg.present_choice_type)
                elif arg.type is not None:
                    new_node = ObjectChoiceNode(arg.type)
                else:
                    continue
                current_leaf.set_arg_value(arg.name, new_node)
                self._predict_step(x_query, new_node, root_y_type,
                                   use_only_train_data, current_depth + 1)
        else:
            raise ValueError(f"leaf node {current_leaf} not predictable")

    def _train_obj_node_step(
        self,
        x_query: str,
        expected: ObjectNodeSet,
        current_gen_root: ObjectChoiceNode,
        current_gen_leaf: ObjectNode,
        teacher_force_path: ObjectNode,
        current_depth: int
    ):
        arg_set_data = expected.get_arg_set_data(teacher_force_path.as_childless_node())
        assert arg_set_data is not None, "Teacher force path not in expected ast set!"
        for arg in teacher_force_path.implementation.children:
            if arg.type is None:
                continue
            next_choice_set = arg_set_data.arg_to_choice_set[arg.name]
            # TODO (DNGros): This is currently somewhat gross as it relies on the _train_step
            # call mutating state. Once it is changed to make changes on current_gen_root
            # this shouldn't be an issue.
            next_gen_leaf = ObjectChoiceNode(arg.next_choice_type)
            current_gen_leaf.set_arg_value(arg.name, next_gen_leaf)
            self._train_step(x_query, next_choice_set, current_gen_root, next_gen_leaf,
                             teacher_force_path.get_choice_node_for_arg(arg.name), current_depth+1)

    def _train_step(
        self,
        x_query: str,
        expected: AstObjectChoiceSet,
        current_gen_root: ObjectChoiceNode,
        current_gen_leaf: ObjectChoiceNode,
        teacher_force_path: ObjectChoiceNode,
        current_depth: int
    ):  # TODO This should likely eventually return the new current_gen ast
        self.type_predictor.train(x_query, current_gen_leaf, expected, current_depth)
        # figure out where going next
        next_expected_node = expected.get_next_node_for_choice(
            teacher_force_path.get_chosen_impl_name())
        assert next_expected_node is not None, "Teacher force path not in expected ast set!"
        next_object_node = ObjectNode(teacher_force_path.next_node.implementation)
        current_gen_leaf.set_choice(next_object_node)
        self._train_obj_node_step(x_query, next_expected_node, current_gen_root,
                                  next_object_node, teacher_force_path.next_node,
                                  current_depth+1)

    def train(
        self,
        x_string: str,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ) -> None:
        current_gen_node = ObjectChoiceNode(teacher_force_path.type_to_choose)
        self._train_step(
            x_string, y_ast, current_gen_node, current_gen_node, teacher_force_path, 0
        )


def _get_type_choice_nodes(
    root_node: AstNode,
    type_name: str
) -> List[Tuple[ObjectChoiceNode, int]]:
    """Gets all the ObjectChoiceNodes that match a certain type inside an AST
    Args:
        root_node: the ast we want to search in
        type_name: the type name choice we are looking for
    Returns:
        List of tuples.
        (A ObjectChoiceNode that is a child of root_node with desired type,
         An int for its depth)
    """
    out = []

    def check_type(cur_node: AstNode, depth: int):
        if cur_node is None:
            return
        if isinstance(cur_node, ObjectChoiceNode):
            if cur_node.get_type_to_choose_name() == type_name:
                out.append((cur_node, depth))
        for child in cur_node.get_children():
            check_type(child, depth + 1)
    check_type(root_node, 0)
    return out


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
        self.parser = StringParser(self.type_context)

    def compare_example(
        self,
        x_query: str,
        current_leaf: ObjectChoiceNode,
        example: Example,
        current_depth: int
    ) -> 'ComparerResult':
        example_ast = self.parser.create_parse_tree(example.ytext, example.ytype)
        return self.comparer.compare(x_query, current_leaf, current_depth,
                                     example.xquery, example_ast)

    def _train_compare(
        self,
        x_query: str,
        current_leaf: ObjectChoiceNode,
        example_to_compare: Example,
        ground_truth_set: AstObjectChoiceSet,
        current_depth: int
    ):
        # Figure out the expected comparer result
        # TODO (DNGros): think about if can memoize during training
        example_ast = self.parser.create_parse_tree(
            example_to_compare.ytext, example_to_compare.ytype)
        choices_in_this_example = _get_type_choice_nodes(
            example_ast, current_leaf.type_to_choose)
        # Extract all potential choices in our ground truth set
        right_choices = [e for e, depth in choices_in_this_example
                         if ground_truth_set.is_known_choice(e.get_chosen_impl_name())]
        in_this_example_impl_name_set = {c.get_chosen_impl_name()
                                         for c, depth in choices_in_this_example}
        right_choices_impl_name_set = {c.get_chosen_impl_name() for c in right_choices}
        this_example_right_prob = 1 if len(right_choices) > 0 else 0
        # Set expected scores. 1 if its valid, otherwise 0
        expected_impl_scores = [(1 if impl_name in right_choices else 0, impl_name)
                                for impl_name in in_this_example_impl_name_set]
        expected_impl_scores.sort()
        expected_result = ComparerResult(this_example_right_prob, tuple(expected_impl_scores))
        self.comparer.train(x_query, current_leaf, current_depth,
                            example_to_compare.xquery, example_ast, expected_result)

    # TODO (DNGros): make a generator
    def _search(
        self,
        x_query,
        current_leaf: ObjectChoiceNode,
        use_only_training_data: bool
    ) -> List[Example]:
        type_name = current_leaf.get_type_to_choose_name()
        split_filter = (DataSplits.TRAIN,) if use_only_training_data else None
        return list(self.index.get_nearest_examples(
            x_query, choose_type_name=type_name, filter_splits=split_filter))

    def predict(
        self,
        x_query: str,
        current_leaf: ObjectChoiceNode,
        use_only_train_data: bool,
        current_depth: int
    ) -> AInixObject:
        if current_depth > 20:
            raise ValueError("whoah, that's too deep man")
        search_results = self._search(x_query, current_leaf, use_only_train_data)
        if not search_results:
            raise ModelCantPredictException(f"No examples in index for '{x_query}'")
        comparer_result = self.compare_example(x_query, current_leaf,
                                               search_results[0], current_depth)
        choose_name = comparer_result.impl_scores[0][1]
        return self.type_context.get_object_by_name(choose_name)

    def train(
        self,
        x_query: str,
        current_leaf: ObjectChoiceNode,
        expected_choices: AstObjectChoiceSet,
        current_depth: int
    ) -> None:
        search_results = self._search(x_query, current_leaf, True)
        if not search_results:
            return
        for result in search_results[:1]:
            self._train_compare(x_query, current_leaf, result, expected_choices, current_depth)


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
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
        expected_result: ComparerResult
    ):
        pass



class SimpleRulebasedComparer(Comparer):
    def compare(
        self,
        gen_query: str,
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
    ) -> ComparerResult:
        potential_type_choice_nodes = _get_type_choice_nodes(
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
        gen_ast_current_leaf: ObjectChoiceNode,
        current_gen_depth: int,
        example_query: str,
        example_ast_root: AstNode,
        expected_result: ComparerResult
    ):
        # RuleBasedComparer doesn't do any training
        pass
