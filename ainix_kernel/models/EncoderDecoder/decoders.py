from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, Callable, List, Optional
import torch
from torch import nn
import attr
from ainix_common.parsing.parseast import ObjectChoiceNode, AstObjectChoiceSet, ObjectNode, AstNode, \
    ObjectNodeSet
from ainix_common.parsing.typecontext import AInixType, AInixObject
from ainix_kernel.model_util.vectorizers import VectorizerBase
from ainix_kernel.model_util.vocab import Vocab, are_indices_valid
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
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ) -> torch.Tensor:
        """
        A forward function to call during training.

        Args:
            query_summary: Tensor of dims (batch, input_dims) which represents
                vector summary representation of all queries in a batch.
            query_encoded_tokens: Tensor of dim (batch, query_len, hidden_size)
                which is an contextual embedding of all tokens in the query.
            y_ast: the ground truth set
            teacher_force_path: The path to take down this the ground truth set

        Returns:
        """
        raise NotImplemented()


class TreeRNNCell(nn.Module):
    """An rnn cell in a tree RNN"""
    def __init__(self, ast_node_embed_size: int, hidden_size):
        super().__init__()
        self.input_size = ast_node_embed_size
        #self.rnn = nn.LSTMCell(ast_node_embed_size, hidden_size)
        self.rnn = nn.GRUCell(ast_node_embed_size, hidden_size)
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
        #out, next_hidden = self.rnn(type_to_predict_features,
        #                            (parent_node_features, last_hidden))
        out = self.rnn(type_to_predict_features, last_hidden)
        #return out, next_hidden
        return out, out


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
        ast_vocab: Vocab,
        bce_pos_weight=1.0
    ):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.object_selector = object_selector
        self.ast_vectorizer = ast_vectorizer
        self.ast_vocab = ast_vocab
        self.bce_pos_weight = bce_pos_weight

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

        predicted_impl = self._predict_most_likely_implementation(
            vectors_to_select_on=outs,
            types_to_select=[current_leaf.type_to_choose]
        )
        assert len(predicted_impl) == len(outs)

        new_node = ObjectNode(predicted_impl[0])
        current_leaf.set_choice(new_node)
        hiddens = self._inference_object_step(new_node, hiddens, cur_depth + 1)
        return hiddens

    def _train_objectchoice_step(
        self,
        last_hidden: torch.Tensor,
        parent_node_features: Optional[torch.Tensor],
        expected: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outs, hiddens = self.rnn_cell(
            last_hidden=last_hidden,
            type_to_predict_features=self._get_ast_node_vectors(teacher_force_path),
            parent_node_features=parent_node_features,
            parent_node_hidden=None,
        )

        loss = self._training_get_loss_from_select_vector(
            vectors_to_select_on=outs,
            types_to_select=[teacher_force_path.type_to_choose],
            current_gt_set=expected
        )

        next_expected_set = expected.get_next_node_for_choice(
            impl_name_chosen=teacher_force_path.get_chosen_impl_name()
        )
        assert next_expected_set is not None, "Teacher force path not in expected ast set!"
        next_object_node = teacher_force_path.next_node
        hiddens, child_loss = self._train_objectnode_step(
            hiddens, next_expected_set, next_object_node)
        return hiddens, loss + child_loss

    def _inference_object_step(
        self,
        current_leaf: ObjectNode,
        last_hidden: Optional[torch.Tensor],
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
                new_node, latest_hidden, my_features, cur_depth + 1)
        return latest_hidden

    def _train_objectnode_step(
        self,
        last_hidden: torch.Tensor,
        expected: ObjectNodeSet,
        teacher_force_path: ObjectNode
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
                latest_hidden, my_features, next_choice_set, next_force_path)
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
        impls_indices = impls_indices[0, :]
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
        #first_prediction = PredictionState(root_type, query_summary)
        #predict_stack: List[PredictionState] = [first_prediction]
        #while predict_stack:
        #    pass
        if is_train:
            if y_ast is None or teacher_force_path is None:
                raise ValueError("If training expect path to be previded")
            last_hidden, loss = self._train_objectchoice_step(
                query_summary, None, y_ast, teacher_force_path)
            return None, loss
        else:
            # TODO (DNGros): make steps this not mutate state and iterative
            root_node = ObjectChoiceNode(root_type)
            last_hidden = self._inference_objectchoice_step(
                root_node, query_summary, None, 0)
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
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ) -> torch.Tensor:
        if not self.training:
            raise ValueError("Expect to be in training mode")
        root_type = y_ast.type_to_choose
        prediction, loss = self.forward(
            query_summary, memory_encoding, root_type, True, y_ast, teacher_force_path)
        return loss

    def _training_get_loss_from_select_vector(
        self,
        vectors_to_select_on: torch.Tensor,
        types_to_select: List[AInixType],
        current_gt_set: AstObjectChoiceSet,
    ) -> torch.Tensor:
        assert len(types_to_select) == 1, "No batch yet"
        assert types_to_select[0] == current_gt_set.type_to_choose
        impls_indices, scores = self.object_selector(vectors_to_select_on, types_to_select)
        impls_indices_correct = are_indices_valid(impls_indices, self.ast_vocab, current_gt_set)
        loss = 0
        for correct_indicies, predicted_score in zip(impls_indices_correct, scores):
            loss += F.binary_cross_entropy_with_logits(
                predicted_score, correct_indicies,
                pos_weight=self.bce_pos_weight
            )
        return loss


def get_default_decoder(
    ast_vocab: Vocab,
    ast_vectorizer: VectorizerBase,
    rnn_hidden_size: int
) -> TreeDecoder:
    rnn_cell = TreeRNNCell(ast_vectorizer.feature_len(), rnn_hidden_size)
    object_selector = objectselector.get_default_object_selector(ast_vocab, ast_vectorizer)
    return TreeRNNDecoder(rnn_cell, object_selector, ast_vectorizer, ast_vocab)
