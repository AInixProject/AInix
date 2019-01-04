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
from ainix_kernel.models.EncoderDecoder.actionselector import ActionSelector, CopyAction, \
    ProduceObjectAction
from ainix_kernel.models.EncoderDecoder.nonretrieval import SimpleActionSelector
from ainix_kernel.models.EncoderDecoder.objectselector import ObjectSelector
from ainix_kernel.models.model_types import ModelException, ModelSafePredictError
import numpy as np
import torch.nn.functional as F

from ainix_kernel.models.multiforward import MultiforwardTorchModule, add_hooks


class TreeDecoder(MultiforwardTorchModule, ABC):
    """An interface for a nn.Module that takes in an query encoding and
    outputs a AST of a certain type."""
    def __init__(self):
        super().__init__()

    @add_hooks
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

    @add_hooks
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
        # TODO (DNGros): Use parent hidden data
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
        action_selector: ActionSelector,
        ast_vectorizer: VectorizerBase,
        ast_vocab: Vocab#,
        #bce_pos_weight=1.0
    ):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.action_selector = action_selector
        self.ast_vectorizer = ast_vectorizer
        self.ast_vocab = ast_vocab

    def _node_to_token_type(self, node: AstNode):
        if isinstance(node, ObjectNode):
            return node.implementation
        elif isinstance(node, ObjectChoiceNode):
            return node.type_to_choose
        raise ValueError("Unrecognized node")

    def _get_ast_node_vectors(self, node: AstNode) -> torch.Tensor:
        # TODO (DNGros): Cache during current train component
        # TODO (DNGros): split ast_vocab into type vocab and object vocab
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

        predicted_action = self.action_selector.infer_predict(
            outs, memory_tokens, current_leaf.type_to_choose)
        if isinstance(predicted_action, CopyAction):
            current_leaf.set_choice(CopyNode(
                current_leaf.type_to_choose, predicted_action.start, predicted_action.end))
        elif isinstance(predicted_action, ProduceObjectAction):
            new_node = ObjectNode(predicted_action.implementation)
            current_leaf.set_choice(new_node)
            hiddens = self._inference_object_step(new_node, hiddens, memory_tokens, cur_depth + 1)
        else:
            raise ValueError()
        return hiddens

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

        loss = self.action_selector.forward_train(
            latent_vec=outs,
            memory_tokens=memory_tokens,
            types_to_select=[teacher_force_path.type_to_choose],
            expected=expected,
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

    @add_hooks
    def forward_predict(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        root_type: AInixType,
    ) -> ObjectChoiceNode:
        if self.training:
            raise ValueError("Expect to not being in training mode during inference.")
        # TODO (DNGros): make steps this not mutate state and iterative
        prediction_root_node = ObjectChoiceNode(root_type)
        last_hidden = self._inference_objectchoice_step(
            prediction_root_node, query_summary, None, memory_encoding, 0)
        return prediction_root_node

    @add_hooks
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
            if y_asts[i] is None or teacher_force_paths[i] is None:
                raise ValueError("If training expect path to be previded")
            last_hidden, loss = self._train_objectchoice_step(
                query_summary[i:i+1], memory_encoding[i:i+1], None, y_asts[i],
                teacher_force_paths[i], 0
            )
            batch_loss += loss
        return batch_loss

    def get_latent_select_states(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        force_path: ObjectChoiceNode
    ) -> Tuple[List[torch.Tensor], List[int]]:
        last_hidden = query_summary
        parent_node_features = None
        parent_node_hidden = None
        #node_to_latent = {}
        latents = []
        y_inds = []
        for y_ind, pointer in enumerate(force_path.depth_first_iter()):
            if isinstance(pointer.cur_node, ObjectChoiceNode):
                parent_features = None if pointer.parent is None \
                    else self._get_ast_node_vectors(pointer.parent.curr_node)
                out, last_hidden = self.rnn_cell(
                    last_hidden=last_hidden,
                    type_to_predict_features=self._get_ast_node_vectors(pointer.cur_node),
                    parent_node_features=parent_features,
                    parent_node_hidden=parent_node_hidden
                )
                latents.append(out)
                y_inds.append(y_ind)
            elif isinstance(pointer.cur_node, ObjectNode):
                pass
            else:
                raise ValueError("Node not recognized.")
        return latents, y_inds


    def get_save_state_dict(self) -> Dict:
        return {
            "version": 0,
            "name": "TreeRNNDecoder",
            "rnn_cell": self.rnn_cell,
            "action_selector": self.action_selector.get_save_state_dict(),
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
        ast_vocab = TypeContextWrapperVocab.create_from_save_state_dict(
            state_dict['ast_vocab'], new_type_context)
        instance = cls(
            rnn_cell=state_dict['rnn_cell'],
            action_selector=SimpleActionSelector.create_from_save_state_dict(
                state_dict['action_selector'], ast_vocab),
            ast_vectorizer=vectorizer_from_save_dict(state_dict['ast_vectorizer']),
            ast_vocab=ast_vocab
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
    action_selector = SimpleActionSelector(rnn_cell.input_size,
        objectselector.get_default_object_selector(ast_vocab, ast_vectorizer), ast_vocab)
    return TreeRNNDecoder(rnn_cell, action_selector, ast_vectorizer, ast_vocab)
