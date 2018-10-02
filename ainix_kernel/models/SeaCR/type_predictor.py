import random
from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
import torch
from ainix_kernel.indexing.exampleindex import ExamplesIndex
from ainix_kernel.indexing.examplestore import Example, DataSplits
from ainix_kernel.models.SeaCR.comparer import ComparerResult
from ainix_kernel.models.model_types import ModelCantPredictException
from ainix_common.parsing.parseast import ObjectChoiceNode, AstObjectChoiceSet, StringParser
from ainix_common.parsing.typecontext import AInixObject
from ainix_kernel.models.SeaCR.comparer import _create_gt_compare_result


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
        self.train_sample_count = 20
        self.train_search_sample_dropout = 0.6
        self.max_examples_to_compare = 10
        self.optimizer = None

    def _create_torch_trainers(self):
        params = self.comparer.get_parameters()
        if params:
            import torch
            self.optimizer = torch.optim.Adam(params, lr=1e-3)
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
            x_query, current_root, current_leaf, current_depth,
            example_to_compare.xquery, example_ast, expected_result)
        if predicted_result is None:
            return None
        #print("pred result", predicted_result)
        #print("example result", expected_result.prob_valid_in_example)
        #expected_tensor = torch.Tensor([[1]])
        expected_tensor = torch.Tensor([[expected_result.prob_valid_in_example]])
        #print("compare", x_query, " y ", example_to_compare.xquery)
        #print("expected tensor", expected_tensor)
        loss = self.present_pred_criterion(
            predicted_result, expected_tensor)
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
            x_query, choose_type_name=type_name, filter_splits=split_filter,
            max_results=self.max_examples_to_compare))

    def _pick_best_choice_from_many_examples(
        self,
        examples: Iterable[Example],
        x_query,
        current_root,
        current_leaf,
        current_depth: int
    ) -> str:
        """Looks at many examples (likely coming from search results), and compares
        the ones that look interesting. It then returns the best type choice to make"""
        # Right now just simply loop through a fixed number and return the one
        # with highest present probability
        # TODO (DNGros): Make this better. Take into account the impl scores and early quitting
        highest_present_prob = 0
        best_choice: str = None
        for example in examples:
            comparer_result = self.compare_example(x_query, current_root, current_leaf,
                                                   example, current_depth)
            if comparer_result.impl_scores is None:
                continue
            if comparer_result.prob_valid_in_example > highest_present_prob:
                best_choice = comparer_result.impl_scores[0][1]
                highest_present_prob = comparer_result.prob_valid_in_example
        if best_choice is None:
            raise ModelCantPredictException(
                f"Comparer did not predict any search results as even potentially valid")
        return best_choice

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
        chosen_name = self._pick_best_choice_from_many_examples(
            search_results, x_query, current_root, current_leaf, current_depth)
        return self.type_context.get_object_by_name(chosen_name)

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
                # also normalize by number of internal comparers
                loss += instance_loss
        # Average loss based off number we actually sampled
        loss /= num_to_sample

        # post training optim step if needed
        if self.optimizer and loss.requires_grad:
            #print("x_query", x_query)
            #print("LOSS", loss)
            loss.backward()
            self.optimizer.step()


class TerribleSearchTypePredictor(SearchingTypePredictor):
    """A version of the search type predictor but it just returns all the
    documents while searching (useful in tests when just trying to test
    the comparer)."""
    def __init__(self, index: ExamplesIndex, comparer: 'Comparer'):
        super().__init__(index, comparer)
        self.train_sample_count = 9e9
        self.train_search_sample_dropout = 0

    def _search(
        self,
        x_query,
        current_leaf: ObjectChoiceNode,
        use_only_training_data: bool
    ):
        splits = (DataSplits.TRAIN,) if use_only_training_data else None
        return list(self.index.get_all_examples(filter_splits=splits))
