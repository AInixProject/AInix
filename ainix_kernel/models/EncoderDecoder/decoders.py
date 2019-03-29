from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, Callable, List, Optional, Dict, Union
import torch
from torch import nn
import attr
from ainix_common.parsing.ast_components import ObjectChoiceNode, AstObjectChoiceSet, \
    ObjectNode, AstNode, ObjectNodeSet, CopyNode
from ainix_common.parsing.model_specific import parse_constants
from ainix_common.parsing.model_specific.tokenizers import StringTokenizer, ModifiedStringToken, \
    get_text_from_tok
from ainix_common.parsing.stringparser import AstUnparser, StringParser
from ainix_common.parsing.typecontext import AInixType, AInixObject, TypeContext
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.model_util import vectorizers
from ainix_kernel.model_util.attending import attend
from ainix_kernel.model_util.vectorizers import VectorizerBase, vectorizer_from_save_dict
from ainix_kernel.model_util.vocab import Vocab, are_indices_valid, TypeContextWrapperVocab
from ainix_kernel.models.EncoderDecoder import objectselector
from ainix_kernel.models.EncoderDecoder.actionselector import ActionSelector, CopyAction, \
    ProduceObjectAction, PathForceSpySelector
from ainix_kernel.models.EncoderDecoder.latentstore import make_latent_store_from_examples
from ainix_kernel.models.EncoderDecoder.nonretrieval import SimpleActionSelector
from ainix_kernel.models.EncoderDecoder.objectselector import ObjectSelector
from ainix_kernel.models.EncoderDecoder.retrieving import RetrievalActionSelector
from ainix_kernel.models.model_types import ModelException, ModelSafePredictError, \
    TypeTranslatePredictMetadata
import numpy as np
import torch.nn.functional as F

from ainix_kernel.models.multiforward import MultiforwardTorchModule, add_hooks
from ainix_kernel.training.augmenting.replacers import Replacer


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
        actual_tokens: List[List[ModifiedStringToken]],
        root_type: AInixType
    ) -> Tuple[ObjectChoiceNode, TypeTranslatePredictMetadata]:
        """
        A forward function to call during inference.

        Args:
            query_summary: Tensor of dims (batch, input_dims) which represents
                vector summary representation of all queries in a batch.
            query_encoded_tokens: Tensor of dim (batch, query_len, hidden_size)
                which is an contextual embedding of all tokens in the query.
            actual_tokens: a list of the actual tokens before they were encoded.
                This is used for stuff like knowing what is a valid start or
                end of a copy.
            root_type: Root type to output

        Returns:
        """
        raise NotImplemented()

    @add_hooks
    def forward_train(
        self,
        query_summary: torch.Tensor,
        query_encoded_tokens: torch.Tensor,
        actual_tokens: List[List[ModifiedStringToken]],
        y_asts: List[AstObjectChoiceSet],
        teacher_force_paths: List[ObjectChoiceNode],
        example_ids: List[int]
    ) -> torch.Tensor:
        """
        A forward function to call during training.

        Args:
            query_summary: Tensor of dims (batch, input_dims) which represents
                vector summary representation of all queries in a batch.
            query_encoded_tokens: Tensor of dim (batch, query_len, hidden_size)
                which is an contextual embedding of all tokens in the query.
            actual_tokens: a list of the actual tokens before they were encoded.
                This is used for stuff like knowing what is a valid start or
                end of a copy.
            y_asts: the ground truth set
            teacher_force_paths: The path to take down this the ground truth set
            example_ids: The ids we are training on. Needed for stuff like
                training a latent store during training.

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
        new_example_store: ExamplesStore,
        replacer: Replacer,
        parser: StringParser,
        unparser: AstUnparser
    ):
        raise NotImplemented

    def get_latent_select_states(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        actual_tokens: List[List[ModifiedStringToken]],
        force_path: ObjectChoiceNode
    ) -> List[torch.Tensor]:
        raise NotImplemented()

    def start_train_session(self):
        pass

    def end_train_session(self):
        pass


class TreeRNNCell(nn.Module):
    """An rnn cell in a tree RNN"""
    def __init__(self, ast_node_embed_size: int, hidden_size):
        super().__init__()
        self.input_size = ast_node_embed_size
        self.rnn = nn.LSTMCell(ast_node_embed_size, hidden_size)
        # self.rnn = nn.GRUCell(ast_node_embed_size, hidden_size)
        self.root_node_features = nn.Parameter(torch.rand(hidden_size))
        self.dropout = nn.Dropout(p=0.1)

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
        next_hidden = self.dropout(next_hidden)
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
    MAX_DEPTH = 50

    def __init__(
        self,
        rnn_cell: TreeRNNCell,
        action_selector: ActionSelector,
        type_vectorizer: VectorizerBase,
        type_context: TypeContext#,
        #bce_pos_weight=1.0
    ):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.action_selector = action_selector
        self.type_vectorizer = type_vectorizer
        self.type_context = type_context

    def _node_to_token_type(self, node: AstNode):
        if isinstance(node, ObjectNode):
            return node.implementation
        elif isinstance(node, ObjectChoiceNode):
            return node.type_to_choose
        raise ValueError("Unrecognized node")

    def _get_obj_choice_features(self, node: ObjectChoiceNode) -> torch.Tensor:
        # TODO (DNGros): Cache during current train component
        indxs = node.type_to_choose.ind
        return self.type_vectorizer(torch.LongTensor([[indxs]]))[:, 0]

    def _inference_objectchoice_step(
        self,
        current_leaf: ObjectChoiceNode,
        last_hidden: torch.Tensor,
        parent_node_features: Optional[torch.Tensor],
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor,
        cur_depth: int,
        override_action_selector: ActionSelector = None
    ) -> Tuple[torch.Tensor, TypeTranslatePredictMetadata]:
        if cur_depth > self.MAX_DEPTH:
            raise ModelException()
        outs, hiddens = self.rnn_cell(
            last_hidden=last_hidden,
            type_to_predict_features=self._get_obj_choice_features(current_leaf),
            parent_node_features=parent_node_features,
            parent_node_hidden=None
        )
        if len(outs) != 1:
            raise NotImplemented("Batches not implemented")

        use_selector = override_action_selector or self.action_selector
        predicted_action, my_metad = use_selector.infer_predict(
            outs, memory_tokens, valid_for_copy_mask,current_leaf.type_to_choose)
        if isinstance(predicted_action, CopyAction):
            current_leaf.set_choice(CopyNode(
                current_leaf.type_to_choose, predicted_action.start, predicted_action.end))
        elif isinstance(predicted_action, ProduceObjectAction):
            new_node = ObjectNode(predicted_action.implementation)
            current_leaf.set_choice(new_node)
            hiddens, child_metad = self._inference_object_step(
                new_node, outs, hiddens, memory_tokens, valid_for_copy_mask,
                cur_depth + 1, override_action_selector)
            my_metad = my_metad.concat(child_metad)
        else:
            raise ValueError()
        return hiddens, my_metad

    def _train_objectchoice_step(
        self,
        last_hidden: torch.Tensor,
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor,
        parent_node_features: Optional[torch.Tensor],
        expected: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode,
        num_parents_with_a_copy_option: int,
        example_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outs, hiddens = self.rnn_cell(
            last_hidden=last_hidden,
            type_to_predict_features=self._get_obj_choice_features(teacher_force_path),
            parent_node_features=parent_node_features,
            parent_node_hidden=None,
        )

        loss = self.action_selector.forward_train(
            latent_vec=outs,
            memory_tokens=memory_tokens,
            valid_for_copy_mask=valid_for_copy_mask,
            types_to_select=[teacher_force_path.type_to_choose],
            expected=expected,
            num_of_parents_with_copy_option=num_parents_with_a_copy_option,
            example_inds=[example_id]
        )

        next_expected_set = expected.get_next_node_for_choice(
            impl_name_chosen=teacher_force_path.get_chosen_impl_name()
        ).next_node
        assert next_expected_set is not None, "Teacher force path not in expected ast set!"
        next_object_node = teacher_force_path.next_node_not_copy
        hiddens, child_loss = self._train_objectnode_step(
            outs, hiddens, memory_tokens, valid_for_copy_mask, next_expected_set, next_object_node,
            num_parents_with_a_copy_option + (1 if expected.copy_is_known_choice() else 0),
            example_id)
        return hiddens, loss + child_loss

    def _inference_object_step(
        self,
        current_leaf: ObjectNode,
        my_features: torch.Tensor,
        last_hidden: Optional[torch.Tensor],
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor,
        cur_depth,
        override_selector: ActionSelector = None
    ) -> Tuple[torch.Tensor, TypeTranslatePredictMetadata]:
        """makes one step for ObjectNodes. Returns last hidden state"""
        if cur_depth > self.MAX_DEPTH:
            raise ModelSafePredictError("Max length exceeded")
        latest_hidden = last_hidden
        metad = TypeTranslatePredictMetadata.create_empty()
        for arg in current_leaf.implementation.children:
            if arg.next_choice_type is not None:
                new_node = ObjectChoiceNode(arg.next_choice_type)
            else:
                continue
            current_leaf.set_arg_value(arg.name, new_node)
            latest_hidden, child_metad = self._inference_objectchoice_step(
                new_node, latest_hidden, my_features, memory_tokens, valid_for_copy_mask,
                cur_depth + 1, override_selector)
            metad = metad.concat(child_metad)
        return latest_hidden, metad

    def _train_objectnode_step(
        self,
        my_features: torch.Tensor,
        last_hidden: torch.Tensor,
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor,
        expected: ObjectNodeSet,
        teacher_force_path: ObjectNode,
        num_of_parents_with_a_copy_option: int,
        example_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        arg_set_data = expected.get_arg_set_data(teacher_force_path.as_childless_node())
        assert arg_set_data is not None, "Teacher force path not in expected ast set!"
        latest_hidden = last_hidden
        child_loss = 0
        for arg in teacher_force_path.implementation.children:
            next_choice_set = arg_set_data.arg_to_choice_set[arg.name]
            next_force_path = teacher_force_path.get_choice_node_for_arg(arg.name)
            latest_hidden, arg_loss = self._train_objectchoice_step(
                latest_hidden, memory_tokens, valid_for_copy_mask, my_features, next_choice_set,
                next_force_path, num_of_parents_with_a_copy_option, example_id)
            child_loss += arg_loss
        return latest_hidden, child_loss

    @add_hooks
    def forward_predict(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        actual_tokens: List[List[ModifiedStringToken]],
        root_type: AInixType,
        override_action_selector: ActionSelector = None
    ) -> Tuple[ObjectChoiceNode, TypeTranslatePredictMetadata]:
        if self.training:
            raise ValueError("Expect to not being in training mode during inference.")
        # TODO (DNGros): make steps this not mutate state and iterative
        prediction_root_node = ObjectChoiceNode(root_type)
        valid_for_copy_mask = get_valid_for_copy_mask(actual_tokens)
        last_hidden, metad = self._inference_objectchoice_step(
            prediction_root_node, query_summary, None, memory_encoding, valid_for_copy_mask,
            0, override_action_selector)
        return prediction_root_node, metad

    @add_hooks
    def forward_train(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        actual_tokens: List[List[ModifiedStringToken]],
        y_asts: List[AstObjectChoiceSet],
        teacher_force_paths: List[ObjectChoiceNode],
        example_ids: List[int]
    ) -> torch.Tensor:
        if not self.training:
            raise ValueError("Expect to be in training mode")

        valid_for_copy_mask = get_valid_for_copy_mask(actual_tokens)
        # Temp hack while can't batch. Just loop through
        batch_loss = 0
        for i in range(len(y_asts)):
            if y_asts[i] is None or teacher_force_paths[i] is None:
                raise ValueError("If training expect path to be previded")
            last_hidden, loss = self._train_objectchoice_step(
                query_summary[i:i+1], memory_encoding[i:i+1], valid_for_copy_mask[i:i+1],
                None, y_asts[i], teacher_force_paths[i], 0, example_ids[i]
            )
            batch_loss += loss
        return batch_loss

    def get_latent_select_states(
        self,
        query_summary: torch.Tensor,
        memory_encoding: torch.Tensor,
        actual_tokens: List[List[ModifiedStringToken]],
        force_path: ObjectChoiceNode
    ) -> List[torch.Tensor]:
        was_in_traing = self.training
        if was_in_traing:
            self.eval()
        try:
            spy_selector = PathForceSpySelector(force_path)
            self.forward_predict(
                query_summary=query_summary,
                memory_encoding=memory_encoding,
                actual_tokens=actual_tokens,
                root_type=force_path.type_to_choose,
                override_action_selector=spy_selector
            )
            lattents_log = spy_selector.lattents_log
            assert spy_selector.y_inds_log == list(range(0, len(lattents_log)*2, 2)), \
                f"what {spy_selector.y_inds_log}"
            return lattents_log
        finally:
            if was_in_traing:
                self.train()

    def start_train_session(self):
        self.action_selector.start_train_session()

    def end_train_session(self):
        self.action_selector.end_train_session()

    def get_save_state_dict(self) -> Dict:
        return {
            "version": 0,
            "name": "TreeRNNDecoder",
            "rnn_cell": self.rnn_cell,
            "action_selector": self.action_selector.get_save_state_dict(),
            "type_vectorizer": self.type_vectorizer.get_save_state_dict(),
            "my_model_state": self.state_dict()
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict,
        new_type_context: TypeContext,
        new_example_store: ExamplesStore,
        replacer: Replacer,
        parser: StringParser,
        unparser: AstUnparser
    ):
        instance = cls(
            rnn_cell=state_dict['rnn_cell'],
            action_selector=make_action_selector_from_dict(
                state_dict['action_selector'], new_type_context, new_example_store,
                replacer, parser, unparser),
            type_vectorizer=vectorizer_from_save_dict(state_dict['type_vectorizer']),
            type_context=new_type_context
        )
        # Caution, this will probably overwrite any speciallness we do in
        # the custom loading. Need to figure that out when we do that.
        instance.load_state_dict(state_dict['my_model_state'])
        return instance


def get_valid_for_copy_mask(tokens: List[List[Union[ModifiedStringToken, str]]]):
    return torch.tensor([
        [0 if get_text_from_tok(tok) in parse_constants.ALL_SPECIALS else 1
         for tok in batch]
        for batch in tokens
    ])


def make_action_selector_from_dict(
    save_dict: dict,
    new_type_context: TypeContext,
    new_example_store: ExamplesStore,
    replacer: Replacer,
    parser: StringParser,
    unparser: AstUnparser
) -> ActionSelector:
    """Creates a action selector off the serialized form. Looks a the name to make the right one"""
    name = save_dict['name']
    if name == "SimpleActionSelector":
        return SimpleActionSelector.create_from_save_state_dict(save_dict, new_type_context)
    elif name == "RetrievalActionSelector":
        latent_store = make_latent_store_from_examples(
            new_example_store, save_dict['latent_size'], replacer, parser, unparser, splits=None)
        return RetrievalActionSelector(latent_store, new_type_context,
                                       save_dict['retrieve_dropout_p'])
    else:
        raise ValueError(f"unrecognized action selector {name}")


def get_default_nonretrieval_decoder(
    type_context: TypeContext,
    rnn_hidden_size: int
) -> TreeDecoder:
    object_vectorizer = vectorizers.TorchDeepEmbed(type_context.get_object_count(), rnn_hidden_size)
    type_vectorizer = vectorizers.TorchDeepEmbed(type_context.get_type_count(), rnn_hidden_size)
    rnn_cell = TreeRNNCell(rnn_hidden_size, rnn_hidden_size)
    action_selector = SimpleActionSelector(rnn_cell.input_size,
                                           objectselector.get_default_object_selector(
                                               type_context, object_vectorizer), type_context)
    return TreeRNNDecoder(rnn_cell, action_selector, type_vectorizer, type_context)


def get_default_retrieval_decoder(
    type_context: TypeContext,
    rnn_hidden_size: int,
    examples: ExamplesStore,
    replacer: Replacer,
    parser: StringParser,
    unparser: AstUnparser
) -> TreeDecoder:
    type_vectorizer = vectorizers.TorchDeepEmbed(type_context.get_type_count(), rnn_hidden_size)
    rnn_cell = TreeRNNCell(rnn_hidden_size, rnn_hidden_size)
    latent_store = make_latent_store_from_examples(
        examples, rnn_hidden_size, replacer, parser, unparser)
    action_selector = RetrievalActionSelector(latent_store, type_context, 0.25)
    return TreeRNNDecoder(rnn_cell, action_selector, type_vectorizer, type_context)
