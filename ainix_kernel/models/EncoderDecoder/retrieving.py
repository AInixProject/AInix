import random
from typing import List, Optional, Tuple

import torch
from torch.distributions import Bernoulli

from ainix_common.parsing.ast_components import AstObjectChoiceSet
from ainix_common.parsing.typecontext import AInixType, TypeContext
from ainix_kernel.model_util.operations import sparse_groupby_sum
from ainix_kernel.model_util.vocab import are_indices_valid
from ainix_kernel.models.EncoderDecoder.actionselector import ActionSelector, ProduceObjectAction, \
    CopyAction
from ainix_kernel.models.EncoderDecoder.latentstore import LatentStore, COPY_IND, LatentStoreTrainer
from ainix_kernel.models.EncoderDecoder.nonretrieval import CopySpanPredictor, \
    get_copy_depth_discount
import torch.nn.functional as F

from ainix_kernel.models.model_types import TypeTranslatePredictMetadata


class RetrievalActionSelector(ActionSelector):
    def __init__(
        self,
        latent_store: LatentStore,
        type_context: TypeContext,
        retrieve_dropout_p: float = 0.5,
        square_feature_reg: float = 0.08
    ):
        super().__init__()
        self.latent_store = latent_store
        self.retrieve_dropout_p = retrieve_dropout_p
        self.max_query_retrieve_count_train = 50
        self.max_query_retrieve_count_infer = 10
        #self.loss_func = torch.nn.MultiLabelSoftMarginLoss()
        # TODO figure out a better loss
        self.loss_func = torch.nn.BCELoss()
        self.span_predictor = CopySpanPredictor(latent_store.latent_size)
        self.type_context = type_context
        self.is_in_training_session = False
        self.latent_store_trainer: Optional[LatentStoreTrainer] = None
        self.square_feature_reg = square_feature_reg
        #if self.latent_store.allow_gradients:
        #    print("LATENTSTORE PARAMETERS", list(self.latent_store.parameters()))
        #    for i, p in enumerate(self.latent_store.parameters()):
        #        param = torch.nn.Parameter(p.data)
        #        self.register_parameter(f"latent_store_values_{i}", param)

    def infer_predict(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        type_to_select: AInixType
    ) -> Tuple['ActionResult', TypeTranslatePredictMetadata]:
        nearest_datas, similarities = self.latent_store.get_n_nearest_latents(
            latent_vec, type_to_select.ind, max_results=self.max_query_retrieve_count_infer)
        # TODO think about whether need to scale for dropout???
        # TODO ahh this is way diffferent than the loss function
        # TODO is softmax best way to do?
        impl_scores, impl_keys = sparse_groupby_sum(
            F.softmax(similarities, dim=0), nearest_datas.impl_choices)
        max_scores_inds = impl_scores.argmax()
        choice_ind = int(impl_keys[max_scores_inds])
        log_total_conf = float(torch.log(impl_scores.max()))
        choose_copy = choice_ind == COPY_IND
        if choose_copy:
            pred_start, pred_end, cp_log_conf = self.span_predictor.inference_predict_span(
                latent_vec, memory_tokens)[0]
            metad = TypeTranslatePredictMetadata.create_leaf_value(cp_log_conf + log_total_conf)
            return CopyAction(pred_start, pred_end), metad
        else:
            impl = self.type_context.get_object_by_ind(choice_ind)
            metad = TypeTranslatePredictMetadata.create_leaf_value(log_total_conf)
            return ProduceObjectAction(impl), metad

    def forward_train(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        types_to_select: List[AInixType],
        expected: AstObjectChoiceSet,
        num_of_parents_with_copy_option: int,
        example_inds: List[int]
    ) -> torch.Tensor:
        if not self.is_in_training_session:
            raise ValueError("must be in training session to train")
        assert len(types_to_select) == len(example_inds) == 1
        #self.latent_store_trainer.update_value(
        #    types_to_select[0].ind, example_inds[0], dfs_depths[0], latent_vec[0])
        nearest_datas, similarities = self.latent_store.get_n_nearest_latents(
            latent_vec[0], types_to_select[0].ind,
            max_results=self.max_query_retrieve_count_train)
        assert len(similarities) <= self.max_query_retrieve_count_train
        loss = torch.tensor(0.0)
        if len(similarities) > 0:   # This could be false if inside a copy with no examples
            keep_mask = torch.rand(*nearest_datas.impl_choices.shape) > self.retrieve_dropout_p
            if float(torch.sum(keep_mask)) == 0:
                keep_mask[random.randint(0, len(keep_mask) - 1)] = 1
            similarities = similarities[keep_mask]
            impls_chosen = nearest_datas.impl_choices[keep_mask]

            impl_scores, impl_keys = sparse_groupby_sum(
                F.softmax(similarities, dim=0), impls_chosen)
            impls_indices_correct = are_indices_valid(
                impl_keys, self.type_context, expected, COPY_IND)
            # TODO weights
            eps = 1e-7
            impl_scores = impl_scores.clamp(eps, 1-eps)
            if len(impl_scores) > 1:
                loss += self.loss_func(impl_scores.unsqueeze(0), impls_indices_correct.unsqueeze(0))
        if expected.copy_is_known_choice():
            span_pred_loss = self.span_predictor.train_predict_span(
                latent_vec, memory_tokens, types_to_select, expected)
            loss += get_copy_depth_discount(num_of_parents_with_copy_option) * span_pred_loss
        # Do l2_loss
        feature_sq_avg = torch.mean(latent_vec ** 2)
        loss += self.square_feature_reg * feature_sq_avg
        return loss

    def start_train_session(self):
        self.is_in_training_session = True
        #self.latent_store_trainer = self.latent_store.get_default_trainer()

    def end_train_session(self):
        self.is_in_training_session = False
        self.latent_store_trainer = None

    def get_save_state_dict(self):
        return {
            "name": "RetrievalActionSelector",
            "version": 0,
            "latent_size": self.latent_store.latent_size,
            # "latent_store": self.latent_store,
            "retrieve_dropout_p": self.retrieve_dropout_p,
            "square_feature_reg": self.square_feature_reg
            #"model_state": self.state_dict()
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        save_dict: dict,
        type_context: TypeContext,
        latent_store: LatentStore
    ) -> 'RetrievalActionSelector':
        instance = RetrievalActionSelector(
            latent_store,
            type_context,
            save_dict['retrieve_dropout_p']
        )
        #instance.load_state_dict(save_dict['model_state'])
        return instance
