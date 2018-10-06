from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, Callable, List

import torch
from torch import nn
import attr

from ainix_common.parsing.parseast import ObjectChoiceNode, AstObjectChoiceSet, ObjectNode, AstNode
from ainix_common.parsing.typecontext import AInixType, AInixObject
from ainix_kernel.model_util.vectorizers import VectorizerBase
from ainix_kernel.model_util.vocab import Vocab
from ainix_kernel.models.EncoderDecoder.objectselector import ObjectSelector
from ainix_kernel.models.model_types import ModelException


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
    ) -> Tuple[ObjectChoiceNode, torch.Tensor]:
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
        root_type: AInixType,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ) -> Tuple[ObjectChoiceNode, torch.Tensor]:
        """
        A forward function to call during training.

        Args:
            query_summary: Tensor of dims (batch, input_dims) which represents
                vector summary representation of all queries in a batch.
            query_encoded_tokens: Tensor of dim (batch, query_len, hidden_size)
                which is an contextual embedding of all tokens in the query.
            root_type: Root type to output

        Returns:
        """
        raise NotImplemented()


class TreeRNNCell(nn.Module):
    """An rnn cell in a tree RNN"""
    def __init__(self, ast_node_embed_size: int, hidden_size):
        super().__init__()
        self.input_size = ast_node_embed_size
        self.rnn = nn.LSTMCell(ast_node_embed_size, hidden_size)

    def forward(
        self,
        last_hidden: torch.Tensor,
        type_to_predict_features: torch.Tensor,
        parent_node_features: torch.Tensor,
        parent_node_hidden: torch.Tensor,
    ):
        # TODO (DNGros): Use parent data
        next_hidden, out = self.rnn(type_to_predict_features, last_hidden)
        return out, next_hidden


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
        ast_vocab: Vocab
    ):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.object_selector = object_selector
        self.ast_vectorizer = ast_vectorizer
        self.ast_vocab = ast_vocab
        self.start_hidden = nn.Parameter(rnn_cell.)

    def _get_ast_node_vectors(self, node: AstNode) -> torch.Tensor:
        # TODO (DNGros): Cache during current train component
        self.ast_vocab.token_to_index(node)
        return self.ast_vectorizer(torch.LongTensor([[node]]))[:, 0]

    def _do_step(self, current_leaf, last_hidden: torch.Tensor,
                 cur_depth, select_func) -> Tuple[torch.Tensor, torch.Tensor]:
        """makes one step. Returns lowest hidden state"""
        if cur_depth > self.MAX_DEPTH:
            raise ModelException()
        if isinstance(current_leaf, ObjectChoiceNode):
            outs, hiddens = self.rnn_cell(last_hidden, self._get_ast_node_vectors(current_leaf),
                                          None, None)
            predicted_impl, this_loss = select_func(outs)
            new_node = ObjectNode(predicted_impl)
            current_leaf.set_choice(new_node)
            if new_node is not None:
                hiddens, child_loss =  self._do_step(new_node, hiddens, cur_depth + 1,
                                                     select_func)
                this_loss += child_loss
            return hiddens, this_loss
        elif isinstance(current_leaf, ObjectNode):
            new_hidden = last_hidden
            this_loss = 0
            for arg in current_leaf.implementation.children:
                if arg.next_choice_type is not None:
                    new_node = ObjectChoiceNode(arg.next_choice_type)
                else:
                    continue
                current_leaf.set_arg_value(arg.name, new_node)
                new_hidden, arg_loss = self._do_step(new_node, last_hidden, cur_depth + 1,
                                                     select_func)
                this_loss += arg_loss
            return new_hidden, this_loss
        else:
            raise ValueError(f"leaf node {current_leaf} not predictable")

    def forward(
        self,
        query_summary: torch.Tensor,
        memory_tokens: torch.Tensor,
        root_type: AInixType,
        next_select_func
    ) -> Tuple[ObjectChoiceNode, torch.Tensor]:
        #first_prediction = PredictionState(root_type, query_summary)
        #predict_stack: List[PredictionState] = [first_prediction]
        #while predict_stack:
        #    pass
        root_node = ObjectChoiceNode(root_type)
        # TODO (DNGros): make this not mutate state and iterative
        last_hidden, loss = self._do_step(root_node, self.rnn_cell.hidden_size,
                                          0, next_select_func)
        return root_node, loss

    def forward_predict(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        root_type: AInixType,
    ) -> ObjectChoiceNode:
        if self.training:
            raise ValueError("Expect to not being in training mode during inference.")
        return self.forward(
            query_summary, memory_encoding, root_type, self._standard_select_func)[0]

    def forward_train(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        root_type: AInixType,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ) -> Tuple[ObjectChoiceNode, torch.Tensor]:
        if not self.training:
            raise ValueError("Expect to be in training mode")
        selection_func = self._make_teacher_forcing_select_func(y_ast, teacher_force_path)
        return self.forward(query_summary, memory_encoding, root_type, selection_func)

    def _standard_select_func(self):
        raise NotImplemented

    def _make_teacher_forcing_select_func(self, y_ast, teacher_force_path) -> Callable:
        """A closure which makes a select function which always takes the
        correct path as specified during training."""
        raise NotImplemented
