import random
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from ainix_kernel.indexing.exampleindex import ExamplesIndex
from ainix_kernel.indexing.examplestore import Example, DataSplits
from ainix_kernel.models.SeaCR.comparer import ComparerResult
from ainix_kernel.models.SeaCR.treeutil import get_type_choice_nodes
from ainix_kernel.models.model_types import ModelCantPredictException
from ainix_common.parsing.parseast import ObjectChoiceNode, AstObjectChoiceSet, StringParser
from ainix_common.parsing.typecontext import AInixObject


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


class TypePredictor(ABC):
    """Interface for something that can take in a current state of ast
    being constructed and output a type prediction for an ObjectChoice node."""
    @abstractmethod
    def predict(
        self,
        x_query: str,
        current_root: ObjectChoiceNode,
        current_leaf: ObjectChoiceNode,
        use_only_train_data: bool,
        current_depth: int
    ) -> AInixObject:
        pass

    @abstractmethod
    def train(
        self,
        x_query: str,
        current_root: ObjectChoiceNode,
        current_leaf: ObjectChoiceNode,
        expected_choices: AstObjectChoiceSet,
        current_depth: int
    ) -> None:
        pass


class SearchingTypePredictor(TypePredictor):
    def __init__(self, index: ExamplesIndex, comparer: 'Comparer'):
        self.index = index
        self.type_context = index.type_context
        self.comparer = comparer
        self.parser = StringParser(self.type_context)
        self.prepared_trainers = False
        self.present_pred_criterion = torch.nn.BCEWithLogitsLoss()
        self.train_sample_count = 10
        self.train_search_sample_dropout = 0.5
        self.optimizer = None

    def _create_torch_trainers(self):
        params = self.comparer.get_parameters()
        if params:
            import torch
            self.optimizer = torch.optim.Adam(params, lr=1e-2)
        self.prepared_trainers = True

    def compare_example(
        self,
        x_query: str,
        current_root: ObjectChoiceNode,
        current_leaf: ObjectChoiceNode,
        example: Example,
        current_depth: int
    ) -> 'ComparerResult':
        example_ast = self.parser.create_parse_tree(example.ytext, example.ytype)
        return self.comparer.compare(x_query, current_root, current_leaf,
                                     current_depth, example.xquery, example_ast)

    def _train_compare(
        self,
        x_query: str,
        current_root: ObjectChoiceNode,
        current_leaf: ObjectChoiceNode,
        example_to_compare: Example,
        ground_truth_set: AstObjectChoiceSet,
        current_depth: int
    ) -> Optional[torch.Tensor]:
        # TODO (DNGros): think about if can memoize during training
        example_ast = self.parser.create_parse_tree(
            example_to_compare.ytext, example_to_compare.ytype)
        expected_result = _create_gt_compare_result(example_ast, current_leaf,
                                                    ground_truth_set)
        predicted_result = self.comparer.train(
            x_query, current_root,current_leaf, current_depth,
            example_to_compare.xquery, example_ast, expected_result)
        if predicted_result is None:
            return None
        print("pred result", predicted_result)
        print("example result", expected_result.prob_valid_in_example)
        loss = self.present_pred_criterion(
            predicted_result, torch.Tensor(
                [[expected_result.prob_valid_in_example]]))
        print("loss", loss)
        return loss


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
        current_root: ObjectChoiceNode,
        current_leaf: ObjectChoiceNode,
        use_only_train_data: bool,
        current_depth: int
    ) -> AInixObject:
        if current_depth > 20:
            raise ValueError("whoah, that's too deep")
        search_results = self._search(x_query, current_leaf, use_only_train_data)
        if not search_results:
            raise ModelCantPredictException(f"No examples in index for '{x_query}'")
        # TODO (DNGros): Test multiple of the results
        comparer_result = self.compare_example(x_query, current_root, current_leaf,
                                               search_results[0], current_depth)
        if len(comparer_result.impl_scores) == 0:
            print("sh")
        print(comparer_result.impl_scores)
        choose_name = comparer_result.impl_scores[0][1]
        return self.type_context.get_object_by_name(choose_name)

    def train(
        self,
        x_query: str,
        current_root: ObjectChoiceNode,
        current_leaf: ObjectChoiceNode,
        expected_choices: AstObjectChoiceSet,
        current_depth: int
    ) -> None:
        # setup
        if not self.prepared_trainers:
            self._create_torch_trainers()

        loss = torch.tensor(0.0)
        if self.optimizer:
            self.optimizer.zero_grad()

        # actual training part
        search_results = self._search(x_query, current_leaf, True)
        if not search_results:
            return
        num_to_sample = min(len(search_results), self.train_sample_count)
        for result in search_results[:num_to_sample]:
            if random.random() < self.train_search_sample_dropout:
                continue
            instance_loss = self._train_compare(
                x_query, current_root, current_leaf, result,
                expected_choices, current_depth)
            if instance_loss is not None:
                loss += instance_loss

        # post training optim step if needed
        if self.optimizer and loss.requires_grad:
            loss.backward()
            self.optimizer.step()


class TerribleSearchTypePredictor(SearchingTypePredictor):
    """A version of the search type predictor but it just returns all the
    documents while searching (useful in tests when just trying to test
    the comparer)."""
    def __init__(self, index: ExamplesIndex, comparer: 'Comparer'):
        super().__init__(index, comparer)
        self.train_sample_count = 9e9

    def _search(
        self,
        x_query,
        current_leaf: ObjectChoiceNode,
        use_only_training_data: bool
    ):
        return self.index.get_all_examples()