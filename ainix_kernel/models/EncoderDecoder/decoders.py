from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, Callable, List, Optional, Dict
import torch
from torch import nn
import attr
from ainix_common.parsing.ast_components import ObjectChoiceNode, AstObjectChoiceSet, \
    ObjectNode, AstNode, ObjectNodeSet, CopyNode
from ainix_common.parsing.typecontext import AInixType, AInixObject, TypeContext
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.model_util.attending import attend
from ainix_kernel.model_util.vectorizers import VectorizerBase, vectorizer_from_save_dict
from ainix_kernel.model_util.vocab import Vocab, are_indices_valid, TypeContextWrapperVocab
from ainix_kernel.models.EncoderDecoder import objectselector
from ainix_kernel.models.EncoderDecoder.objectselector import ObjectSelector
from ainix_kernel.models.model_types import ModelException, ModelSafePredictError
import numpy as np
import torch.nn.functional as F


class TreeDecoder(nn.Module, ABC):
    """An interface for a nn.Module that takes in an query encoding and
    outputs a AST of a certain type."""
    def __init__(self):
        super().__init__()

    def forward_predict(
        self,
        query_summary: torch.Tensor,
        query_encoded_tokens: torch.Tensor,
        root_type: AInixType
    ) -> ObjectChoiceNode:
        """
        A forward function to call during inference.

        Args:
            query_summary: Tensor of dims (batch, input_dims) which represents
                vector summary representation of all queries in a batch.
            query_encoded_tokens: Tensor of dim (batch, query_len, hidden_size)
                which is an contextual embedding of all tokens in the query.
            root_type: Root type to output

        Returns:
        """
        raise NotImplemented()

    def forward_train(
        self,
        query_summary: torch.Tensor,
        query_encoded_tokens: torch.Tensor,
        y_asts: List[AstObjectChoiceSet],
        teacher_force_paths: List[ObjectChoiceNode]
    ) -> torch.Tensor:
        """
        A forward function to call during training.

        Args:
            query_summary: Tensor of dims (batch, input_dims) which represents
                vector summary representation of all queries in a batch.
            query_encoded_tokens: Tensor of dim (batch, query_len, hidden_size)
                which is an contextual embedding of all tokens in the query.
            y_asts: the ground truth set
            teacher_force_paths: The path to take down this the ground truth set

        Returns: loss
        """
        raise NotImplemented()

    def get_save_state_dict(self) -> Dict:
        raise NotImplemented()

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict,
        new_type_context: TypeContext,
        new_example_store: ExamplesStore
    ):
        raise NotImplemented


class TreeRNNCell(nn.Module):
    """An rnn cell in a tree RNN"""
    def __init__(self, ast_node_embed_size: int, hidden_size):
        super().__init__()
        self.input_size = ast_node_embed_size
        self.rnn = nn.LSTMCell(ast_node_embed_size, hidden_size)
        # self.rnn = nn.GRUCell(ast_node_embed_size, hidden_size)
        self.root_node_features = nn.Parameter(torch.rand(hidden_size))

    def forward(
        self,
        last_hidden: torch.Tensor,
        type_to_predict_features: torch.Tensor,
        parent_node_features: torch.Tensor,
        parent_node_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            last_hidden:
            type_to_predict_features:
            parent_node_features:
            parent_node_hidden:

        Returns:
            Tuple of tensor. First the the thing to predict on. Second is
            internal state to pass forward.
        """
        # TODO (DNGros): Use parent data
        if parent_node_features is None:
            num_of_batches = len(type_to_predict_features)
            parent_node_features = self.root_node_features.expand(num_of_batches, -1)
        out, next_hidden = self.rnn(type_to_predict_features,
                                    (parent_node_features, last_hidden))
        #out = self.rnn(type_to_predict_features, last_hidden)
        return out, next_hidden
        #return out, out


#@attr.s
#class PredictionState:
#    """A struct used to keep track state during sequential prediction on
#    the rnn_cell."""
#    type_to_choose: AInixType
#    parent_object: AInixObject
#    last_hidden_state: torch.Tensor


class TreeRNNDecoder(TreeDecoder):
    MAX_DEPTH = 30

    def __init__(
        self,
        rnn_cell: TreeRNNCell,
        object_selector: ObjectSelector,
        ast_vectorizer: VectorizerBase,
        ast_vocab: Vocab#,
        #bce_pos_weight=1.0
    ):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.object_selector = object_selector
        self.ast_vectorizer = ast_vectorizer
        self.ast_vocab = ast_vocab
        # Copy stuff. Should probably be moved to its own module, but for now
        # I'm being lazy because if switch to retrieval method this will change
        # anyways.

        # copy_relevant_linear is a projection into a space which is shared for
        # all of predicting whether to copy, the start, and the end.
        # It assumed there is shared information about whether to copy or not.
        self.copy_relevant_linear = nn.Sequential(
            nn.Linear(rnn_cell.input_size, rnn_cell.input_size),
            nn.ReLU()
        )
        self.should_copy_predictor = nn.Sequential(
            self.copy_relevant_linear,
            nn.Linear(rnn_cell.input_size, int(rnn_cell.input_size / 4)),
            nn.ReLU(),
            nn.Linear(int(rnn_cell.input_size / 4), 1)
        )
        self.copy_start_vec_predictor = nn.Sequential(
            nn.Linear(rnn_cell.input_size, rnn_cell.input_size),
            nn.ReLU()
        )
        self.copy_end_vec_predictor = nn.Sequential(
            nn.Linear(rnn_cell.input_size, rnn_cell.input_size),
            nn.ReLU()
        )
        # TODO (DNGros): Figure this out. It changed in torch 1.0 and is weird now
        #self.bce_pos_weight = bce_pos_weight

    def _node_to_token_type(self, node: AstNode):
        if isinstance(node, ObjectNode):
            return node.implementation
        elif isinstance(node, ObjectChoiceNode):
            return node.type_to_choose
        raise ValueError("Unrecognized node")

    def _get_ast_node_vectors(self, node: AstNode) -> torch.Tensor:
        # TODO (DNGros): Cache during current train component
        indxs = self.ast_vocab.token_to_index(self._node_to_token_type(node))
        return self.ast_vectorizer(torch.LongTensor([[indxs]]))[:, 0]

    def _inference_objectchoice_step(
        self,
        current_leaf: ObjectChoiceNode,
        last_hidden: torch.Tensor,
        parent_node_features: Optional[torch.Tensor],
        memory_tokens: torch.Tensor,
        cur_depth
    ) -> torch.Tensor:
        if cur_depth > self.MAX_DEPTH:
            raise ModelException()
        outs, hiddens = self.rnn_cell(
            last_hidden=last_hidden,
            type_to_predict_features=self._get_ast_node_vectors(current_leaf),
            parent_node_features=parent_node_features,
            parent_node_hidden=None,
        )
        if len(outs) != 1:
            raise NotImplemented("Batches not implemented")

        copy_probs = self._predict_whether_copy(outs)
        do_copy = copy_probs[0] > 0.5
        if do_copy:
            pred_start, pred_end = self._predict_copy_span(outs, memory_tokens)[0]
            current_leaf.set_choice(CopyNode(current_leaf.type_to_choose, pred_start, pred_end))
        else:
            predicted_impl = self._predict_most_likely_implementation(
                vectors_to_select_on=outs,
                types_to_select=[current_leaf.type_to_choose]
            )
            assert len(predicted_impl) == len(outs)

            new_node = ObjectNode(predicted_impl[0])
            current_leaf.set_choice(new_node)
            hiddens = self._inference_object_step(new_node, hiddens, memory_tokens, cur_depth + 1)
        return hiddens

    def _predict_whether_copy(
        self,
        selection_vector: torch.Tensor,
        to_logit=True
    ) -> torch.Tensor:
        """Returns a value representing whether or not a copy should be chosen.

        Args:
            selection_vector: the vector we use to predict. Of dim (batch, hidden_size)
            to_logit: whether to apply a sigmoid before returning.
        """
        v = self.should_copy_predictor(selection_vector)
        return F.sigmoid(v) if to_logit else v

    def _get_copy_span_weights(
        self,
        select_on_vec: torch.Tensor,
        memory_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a copy is valid, predicts the probability over each token in
        of it being a start or end of a copy.

        Args:
            select_on_vec: A hidden state vector we will use when doing predictions.
                Should be of size (batch, hidden_size)
            memory_tokens: the hidden states of each token to predict on.
                Should be of dim (batch, num_tokens, hidden_dim)
        Returns:
            start_weights: A prediction whether each token is the start of the span.
                Of dim (batch, num_tokens). NOTE: this is not normalized (should pass
                through softmax or log softmax to get actual probability)
            end_weights: Same as start_weights, but for the INCLUSIVE end of the span
        """
        copy_vec = self.copy_relevant_linear(select_on_vec)
        start_vec = self.copy_start_vec_predictor(copy_vec)
        # TODO (DNGros): when batch, should pass in the lengths so can mask right
        start_predictions = attend.get_attend_weights(
            start_vec.unsqueeze(0), memory_tokens, normalize='identity').squeeze(0)
        end_vec = self.copy_end_vec_predictor(copy_vec)
        end_predictions = attend.get_attend_weights(
            end_vec.unsqueeze(0), memory_tokens, normalize='identity').squeeze(0)
        return start_predictions, end_predictions

    def _predict_copy_span(
        self,
        select_on_vec: torch.Tensor,
        memory_tokens: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Given a copy is valid, finds the most likely (start, end) span"""
        start_preds, end_preds = self._get_copy_span_weights(select_on_vec, memory_tokens)
        starts = torch.argmax(start_preds, dim=1)
        ends = torch.argmax(end_preds, dim=1)
        # TODO should eventually also probably return the log probability of this span
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def _train_objectchoice_step(
        self,
        last_hidden: torch.Tensor,
        memory_tokens: torch.Tensor,
        parent_node_features: Optional[torch.Tensor],
        expected: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode,
        num_parents_with_a_copy_option: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outs, hiddens = self.rnn_cell(
            last_hidden=last_hidden,
            type_to_predict_features=self._get_ast_node_vectors(teacher_force_path),
            parent_node_features=parent_node_features,
            parent_node_hidden=None,
        )

        loss = self._training_get_loss_from_select_vector(
            vectors_to_select_on=outs,
            memory_tokens=memory_tokens,
            types_to_select=[teacher_force_path.type_to_choose],
            current_gt_set=expected,
            num_of_parents_with_copy_option=num_parents_with_a_copy_option
        )

        next_expected_set = expected.get_next_node_for_choice(
            impl_name_chosen=teacher_force_path.get_chosen_impl_name()
        ).next_node
        assert next_expected_set is not None, "Teacher force path not in expected ast set!"
        next_object_node = teacher_force_path.next_node_not_copy
        hiddens, child_loss = self._train_objectnode_step(
            hiddens, memory_tokens, next_expected_set, next_object_node,
            num_parents_with_a_copy_option + (1 if expected.copy_is_known_choice() else 0))
        return hiddens, loss + child_loss

    def _inference_object_step(
        self,
        current_leaf: ObjectNode,
        last_hidden: Optional[torch.Tensor],
        memory_tokens: torch.Tensor,
        cur_depth
    ) -> torch.Tensor:
        """makes one step for ObjectNodes. Returns last hidden state"""
        if cur_depth > self.MAX_DEPTH:
            raise ModelSafePredictError("Max length exceeded")
        latest_hidden = last_hidden
        my_features = self._get_ast_node_vectors(current_leaf)
        for arg in current_leaf.implementation.children:
            if arg.next_choice_type is not None:
                new_node = ObjectChoiceNode(arg.next_choice_type)
            else:
                continue
            current_leaf.set_arg_value(arg.name, new_node)
            latest_hidden = self._inference_objectchoice_step(
                new_node, latest_hidden, my_features, memory_tokens, cur_depth + 1)
        return latest_hidden

    def _train_objectnode_step(
        self,
        last_hidden: torch.Tensor,
        memory_tokens: torch.Tensor,
        expected: ObjectNodeSet,
        teacher_force_path: ObjectNode,
        num_of_parents_with_a_copy_option: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        arg_set_data = expected.get_arg_set_data(teacher_force_path.as_childless_node())
        assert arg_set_data is not None, "Teacher force path not in expected ast set!"
        latest_hidden = last_hidden
        child_loss = 0
        my_features = self._get_ast_node_vectors(teacher_force_path)
        for arg in teacher_force_path.implementation.children:
            next_choice_set = arg_set_data.arg_to_choice_set[arg.name]
            next_force_path = teacher_force_path.get_choice_node_for_arg(arg.name)
            latest_hidden, arg_loss = self._train_objectchoice_step(
                latest_hidden, memory_tokens, my_features, next_choice_set,
                next_force_path, num_of_parents_with_a_copy_option)
            child_loss += arg_loss
        return latest_hidden, child_loss

    def _predict_most_likely_implementation(
        self,
        vectors_to_select_on: torch.Tensor,
        types_to_select: List[AInixType]
    ) -> np.ndarray:
        impls_indices, scores = self.object_selector(vectors_to_select_on, types_to_select)
        # Because not batch yet, just take first of each
        assert len(scores) == 1, "No batch yet"
        scores = scores[0]
        impls_indices = impls_indices[0]
        ####
        best_scores = torch.argmax(scores)
        best_obj_indxs = impls_indices[best_scores]
        return self.ast_vocab.torch_indices_to_tokens(torch.stack([best_obj_indxs]))

    def forward(
        self,
        query_summary: torch.Tensor,
        memory_tokens: torch.Tensor,
        root_type: AInixType,
        is_train: bool,
        y_ast: Optional[AstObjectChoiceSet],
        teacher_force_path: Optional[ObjectChoiceNode]
    ) -> Tuple[Optional[ObjectChoiceNode], Optional[torch.Tensor]]:
        if is_train:
            if y_ast is None or teacher_force_path is None:
                raise ValueError("If training expect path to be previded")
            last_hidden, loss = self._train_objectchoice_step(
                query_summary, memory_tokens, None, y_ast, teacher_force_path, 0)
            return None, loss
        else:
            # TODO (DNGros): make steps this not mutate state and iterative
            root_node = ObjectChoiceNode(root_type)
            last_hidden = self._inference_objectchoice_step(
                root_node, query_summary, None, memory_tokens, 0)
            return root_node, None

    def forward_predict(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        root_type: AInixType,
    ) -> ObjectChoiceNode:
        if self.training:
            raise ValueError("Expect to not being in training mode during inference.")
        prediction, loss = self.forward(
            query_summary, memory_encoding, root_type, False, None, None)
        return prediction

    def forward_train(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        y_asts: List[AstObjectChoiceSet],
        teacher_force_paths: List[ObjectChoiceNode]
    ) -> torch.Tensor:
        if not self.training:
            raise ValueError("Expect to be in training mode")

        # Temp hack while can't batch. Just loop through
        batch_loss = 0
        for i in range(len(y_asts)):
            root_type = y_asts[i].type_to_choose
            prediction, loss = self.forward(
                query_summary[i:i+1], memory_encoding[i:i+1], root_type, True,
                y_asts[i], teacher_force_paths[i]
            )
            batch_loss += loss
        return batch_loss

    def _train_loss_from_choose_whether_copy(
        self,
        vectors_to_select_on: torch.Tensor,
        types_to_select: List[AInixType],
        current_gt_set: AstObjectChoiceSet
    ) -> torch.Tensor:
        assert len(types_to_select) == 1, "No batch yet"
        assert types_to_select[0] == current_gt_set.type_to_choose
        should_be_true = torch.Tensor([[1 if current_gt_set.copy_is_known_choice() else 0]])
        predictions = self._predict_whether_copy(vectors_to_select_on, to_logit=False)
        return F.binary_cross_entropy_with_logits(
            predictions,
            should_be_true
        )

    def _train_loss_from_choosing_copy_span(
        self,
        vectors_to_select_on: torch.Tensor,
        memory_tokens: torch.Tensor,
        types_to_select: List[AInixType],
        current_gt_set: AstObjectChoiceSet
    ) -> torch.Tensor:
        if not current_gt_set.copy_is_known_choice():
            return 0

        si, ei = current_gt_set.earliest_known_copy()
        correct_starts = torch.LongTensor([si])
        correct_ends = torch.LongTensor([ei])
        start_predictions, end_predictions = self._get_copy_span_weights(
            vectors_to_select_on, memory_tokens)

        #print(f"start pred {start_predictions} want {correct_starts}")
        #print(f"end pred {end_predictions} want {correct_ends}")

        start_loss = F.cross_entropy(
            start_predictions,
            correct_starts
        )
        end_loss = F.cross_entropy(
            end_predictions,
            correct_ends
        )
        # TODO Weight deeper copy predictions less to not saturate higher ones
        return start_loss + end_loss

    def _training_get_loss_from_select_vector(
        self,
        vectors_to_select_on: torch.Tensor,
        memory_tokens: torch.Tensor,
        types_to_select: List[AInixType],
        current_gt_set: AstObjectChoiceSet,
        num_of_parents_with_copy_option: int
    ) -> torch.Tensor:
        assert len(types_to_select) == 1, "No batch yet"
        assert types_to_select[0] == current_gt_set.type_to_choose
        impls_indices, scores = self.object_selector(vectors_to_select_on, types_to_select)
        assert len(impls_indices) == 1 and len(scores) == 1, "no batch yet"
        impls_indices_correct = are_indices_valid(impls_indices[0], self.ast_vocab, current_gt_set)
        loss = 0
        for correct_indicies, predicted_score in zip(impls_indices_correct, scores[0]):
            loss += F.binary_cross_entropy_with_logits(
                predicted_score, correct_indicies#,
                #pos_weight=self.bce_pos_weight
            )
        loss /= len(scores)
        loss += self._train_loss_from_choose_whether_copy(
            vectors_to_select_on, types_to_select, current_gt_set)
        span_pred_loss = self._train_loss_from_choosing_copy_span(
            vectors_to_select_on, memory_tokens, types_to_select, current_gt_set)
        # What we really care about is getting the top level span prediction correct
        # as ideally we will predict copy at the highest level, and not need to predict
        # the lower spans. Therefore we discount the lower span predictions
        copy_depth_discount = 1 / (1 + 1*num_of_parents_with_copy_option)
        loss += copy_depth_discount * span_pred_loss

        return loss

    def get_save_state_dict(self) -> Dict:
        return {
            "version": 0,
            "name": "TreeRNNDecoder",
            "rnn_cell": self.rnn_cell,
            "object_selector": self.object_selector,
            "ast_vectorizer": self.ast_vectorizer.get_save_state_dict(),
            "ast_vocab": self.ast_vocab.get_save_state_dict(),
            "my_model_state": self.state_dict()
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict,
        new_type_context: TypeContext,
        new_example_store: ExamplesStore
    ):
        instance = cls(
            rnn_cell=state_dict['rnn_cell'],
            object_selector=state_dict['object_selector'],
            ast_vectorizer=vectorizer_from_save_dict(state_dict['ast_vectorizer']),
            ast_vocab=TypeContextWrapperVocab.create_from_save_state_dict(
                state_dict['ast_vocab'], new_type_context)
        )
        # Caution, this will probably overwrite any speciallness we do in
        # the custom loading. Another reason to put the copying in its own module.
        instance.load_state_dict(state_dict['my_model_state'])
        return instance

def get_default_decoder(
    ast_vocab: Vocab,
    ast_vectorizer: VectorizerBase,
    rnn_hidden_size: int
) -> TreeDecoder:
    rnn_cell = TreeRNNCell(ast_vectorizer.feature_len(), rnn_hidden_size)
    object_selector = objectselector.get_default_object_selector(ast_vocab, ast_vectorizer)
    return TreeRNNDecoder(rnn_cell, object_selector, ast_vectorizer, ast_vocab)
