"""Selectors for the nonretrieval method"""
import math

import numpy as np
from abc import ABC
from typing import Tuple, List, Type

import torch

from torch import nn

from ainix_common.parsing.ast_components import ObjectNodeLike, ObjectNode, CopyNode, \
    AstObjectChoiceSet
from ainix_common.parsing.typecontext import AInixType, TypeContext
from ainix_kernel.model_util import vocab
from ainix_kernel.model_util.attending import attend
from ainix_kernel.model_util.vocab import are_indices_valid, Vocab
from ainix_kernel.models.EncoderDecoder.actionselector import ActionSelector, CopyAction, \
    ProduceObjectAction, ActionResult
from ainix_kernel.models.EncoderDecoder.objectselector import ObjectSelector
from ainix_kernel.models.model_types import TypeTranslatePredictMetadata
from ainix_kernel.models.multiforward import add_hooks, MultiforwardTorchModule
import torch.nn.functional as F


class SimpleActionSelector(ActionSelector):
    def __init__(
        self,
        latent_size: int,
        object_selector: ObjectSelector,
        type_context: TypeContext
    ):
        super().__init__()
        self.object_selector = object_selector
        self.type_context = type_context
        self.latent_size = latent_size
        print("latent_size", latent_size)
        # Copy stuff. Should probably be moved to its own module, but for now
        # I'm being lazy because if switch to retrieval method this will change
        # anyways.
        self.should_copy_predictor = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, int(latent_size / 4)),
            nn.ReLU(),
            nn.Linear(int(latent_size / 4), 1)
        )
        self.span_predictor = CopySpanPredictor(latent_size)
        # TODO (DNGros): Figure this out. It changed in torch 1.0 and is weird now
        #self.bce_pos_weight = bce_pos_weight

    @add_hooks
    def infer_predict(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor,
        type_to_select: AInixType
    ) -> Tuple[ActionResult, TypeTranslatePredictMetadata]:
        copy_probs = self._predict_whether_copy(latent_vec)
        do_copy = copy_probs[0] > 0.5
        if do_copy:
            pred_start, pred_end, log_conf = self.span_predictor.inference_predict_span(
                latent_vec, memory_tokens, valid_for_copy_mask)[0]
            full_copy_log_conf = math.log(copy_probs[0]) + log_conf
            metad = TypeTranslatePredictMetadata.create_leaf_value(full_copy_log_conf)
            return CopyAction(pred_start, pred_end), metad
        else:
            predicted_impl, metad = self._predict_most_likely_implementation(
                vectors_to_select_on=latent_vec,
                types_to_select=[type_to_select]
            )
            assert len(predicted_impl) == len(latent_vec)
            return ProduceObjectAction(predicted_impl[0]), metad

    @add_hooks
    def forward_train(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor,
        types_to_select: List[AInixType],
        expected: AstObjectChoiceSet,
        num_of_parents_with_copy_option: int,
        example_inds: List[int]
    ) -> torch.Tensor:
        assert len(types_to_select) == 1, "No batch yet"
        assert types_to_select[0] == expected.type_to_choose
        impls_indices, scores = self.object_selector(latent_vec, types_to_select)
        assert len(impls_indices) == 1 and len(scores) == 1, "no batch yet"
        impls_indices_correct = are_indices_valid(impls_indices[0], self.type_context, expected)
        loss = 0
        for correct_indicies, predicted_score in zip(impls_indices_correct, scores[0]):
            loss += F.binary_cross_entropy_with_logits(
                predicted_score, correct_indicies#,
                #pos_weight=self.bce_pos_weight
            )
        loss /= len(scores)
        loss += self._train_loss_from_choose_whether_copy(
            latent_vec, types_to_select, expected)
        span_pred_loss = self.span_predictor.train_predict_span(
            latent_vec, memory_tokens, valid_for_copy_mask, types_to_select, expected)
        assert num_of_parents_with_copy_option <= 0
        loss += get_copy_depth_discount(num_of_parents_with_copy_option) * span_pred_loss

        return loss

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

    def _predict_most_likely_implementation(
        self,
        vectors_to_select_on: torch.Tensor,
        types_to_select: List[AInixType]
    ) -> Tuple[np.ndarray, TypeTranslatePredictMetadata]:
        impls_indices, scores = self.object_selector(vectors_to_select_on, types_to_select)
        # Because not batch yet, just take first of each
        assert len(scores) == 1, "No batch yet"
        scores = scores[0]
        impls_indices = impls_indices[0]
        ####
        log_softmaxes = F.log_softmax(scores, dim=0)
        best_scores = torch.argmax(scores)
        best_obj_indxs = impls_indices[best_scores]
        metad = TypeTranslatePredictMetadata.create_leaf_value(float(torch.max(log_softmaxes)))
        best_impl = vocab.torch_inds_to_objects(torch.stack([best_obj_indxs]), self.type_context)
        return best_impl, metad


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
        return torch.sigmoid(v) if to_logit else v

    def get_save_state_dict(self) -> dict:
        return {
            "name": "SimpleActionSelector",
            "version": 0,
            "latent_size": self.latent_size,
            "object_selector": self.object_selector,
            #"state_dict": self.state_dict()
        }

    @classmethod
    def create_from_save_state_dict(cls, save_dict: dict, type_context: TypeContext):
        instance = cls(
            latent_size=save_dict['latent_size'],
            object_selector=save_dict['object_selector'],
            type_context=type_context
        )
        #instance.load_state_dict(save_dict['state_dict'])
        return instance


def get_copy_depth_discount(num_of_parents_with_copy_option: int):
    """What we really care about is getting the top level span prediction correct
    as ideally we will predict copy at the highest level, and not need to predict
    the lower spans. Therefore we discount the lower span predictions"""
    return 1 / (1 + 1*num_of_parents_with_copy_option)


class CopySpanPredictor(MultiforwardTorchModule):
    def __init__(self, latent_size: int):
        super().__init__()
        self.copy_relevant_linear = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.PReLU()
        )
        self.copy_start_vec_predictor = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.PReLU()
        )
        self.copy_end_vec_predictor = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.PReLU()
        )

    def inference_predict_span(
        self,
        select_on_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor
    ) -> List[Tuple[int, int, float]]:
        """Given a copy is valid, finds the most likely (start, end, log_confidence) span"""
        start_preds, end_preds = self._get_copy_span_weights(
            select_on_vec, memory_tokens, valid_for_copy_mask)
        starts = torch.argmax(start_preds, dim=1)
        # condition end after start
        # TODO figure out how to batch this
        assert len(select_on_vec) == 1, "No batch support 😞"
        ends_after_start = end_preds[0][starts[0]:].unsqueeze(0)
        ends = starts + torch.argmax(ends_after_start, dim=1)
        confidences = F.log_softmax(start_preds, dim=1)[:, starts] + \
                      F.log_softmax(end_preds, dim=1)[:, ends]
        out = [(int(s), int(e), float(conf[0])) for s, e, conf in zip(starts, ends, confidences)]
        assert len(out) > 0
        return out

    def train_predict_span(
        self,
        vectors_to_select_on: torch.Tensor,
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor,
        types_to_select: List[AInixType],
        current_gt_set: AstObjectChoiceSet
    ) -> torch.Tensor:
        if not current_gt_set.copy_is_known_choice():
            return 0

        si, ei = current_gt_set.earliest_known_copy()
        if si is None or ei is None:
            raise ValueError("No copy in the gt??")
        correct_starts = torch.LongTensor([si])
        correct_ends = torch.LongTensor([ei])
        start_predictions, end_predictions = self._get_copy_span_weights(
            vectors_to_select_on, memory_tokens, valid_for_copy_mask)

        start_loss = F.cross_entropy(
            start_predictions,
            correct_starts
        )
        end_loss = F.cross_entropy(
            end_predictions,
            correct_ends
        )
        return start_loss + end_loss

    def _get_copy_span_weights(
        self,
        select_on_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        valid_for_copy_mask: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a copy is valid, predicts the probability over each token in
        of it being a start or end of a copy.

        Args:
            select_on_vec: A hidden state vector we will use when doing predictions.
                Should be of size (batch, hidden_size)
            memory_tokens: the hidden states of each token to predict on.
                Should be of dim (batch, num_tokens, hidden_dim)
            valid_for_copy_mask: Tensor of dim (batch, num_tokens). For places
                which is a valid start or end to a copy. For example spaces
                or specials might not be a valid thing start or end on.
        Returns:
            start_weights: A prediction whether each token is the start of the span.
                Of dim (batch, num_tokens). NOTE: this is not normalized (should pass
                through softmax or log softmax to get actual probability)
            end_weights: Same as start_weights, but for the INCLUSIVE end of the span
        """
        # Make a big negative value for anything we don't want to take
        mask_addition = (1.0 - valid_for_copy_mask.float()) * -10000.0

        copy_vec = self.copy_relevant_linear(select_on_vec)
        start_vec = self.copy_start_vec_predictor(copy_vec)
        start_predictions = attend.get_attend_weights(
            start_vec.unsqueeze(0), memory_tokens, normalize='identity').squeeze(0)
        end_vec = self.copy_end_vec_predictor(copy_vec)
        end_predictions = attend.get_attend_weights(
            end_vec.unsqueeze(0), memory_tokens, normalize='identity').squeeze(0)
        return start_predictions + mask_addition, end_predictions + mask_addition


#object_chooser: 'ObjectnodeChooser',
#copy_chooser: 'CopySelector',
#action_kind_projector: 'ActionKindSelector',
#action_kind_project_out_size: int

#self.object_chooser = object_chooser
#self.copy_chooser = copy_chooser
#self.action_kind_projector = action_kind_projector


#class CopySelector(MultiforwardTorchModule, ABC):
#    """Given we are doing a copy action, this constructs the proper args"""
#    @add_hooks
#    def infer_predict(self, latent_vec: torch.Tensor) -> 'CopyAction':
#        pass
#
#    @add_hooks
#    def forward_train(
#            self,
#            latent_vec: torch.Tensor,
#            expected: CopyNode
#    ) -> torch.Tensor:
#        pass
#
#
#class ObjectnodeChooser(MultiforwardTorchModule, ABC):
#    """Given we are doing a produce object action, this constructs the proper args"""
#    @add_hooks
#    def infer_predict(self, latent_vec: torch.Tensor) -> 'CopyAction':
#        pass
#
#    @add_hooks
#    def forward_train(
#            self,
#            latent_vec: torch.Tensor,
#            expected: CopyNode
#    ) -> torch.Tensor:
#        pass
#
#
#class ActionKindChooser(nn.Module):
#    def __init__(self, in_size: int):
#        super().__init__()
#        self.action_kinds = (CopyAction, ProduceObjectAction)
#        self.action_kind_embedding = nn.Parameter(
#            torch.rand(len(self.action_kins), in_size / 2))
#        self.action_projector = nn.Sequential(
#            nn.Linear(in_size, in_size),
#            nn.ReLU(),
#            nn.Linear(in_size, in_size / 2)
#        )
#
#    def forward(self, latent_vec: torch.Tensor) -> Tuple[torch.Tensor, List[Type[ActionSelector]]]:
#        projection = self.action_projector(latent_vec)
#        alignments = torch.sum(projection*self.action_kind_embedding, dim=1)

#        F.softmax(alignments, dim=0)
