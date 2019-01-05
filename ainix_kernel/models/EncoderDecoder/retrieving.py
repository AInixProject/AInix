from typing import List

import torch
from torch.distributions import Bernoulli

from ainix_common.parsing.ast_components import AstObjectChoiceSet
from ainix_common.parsing.typecontext import AInixType
from ainix_kernel.model_util.operations import sparse_groupby_sum
from ainix_kernel.model_util.vocab import are_indices_valid
from ainix_kernel.models.EncoderDecoder.actionselector import ActionSelector
from ainix_kernel.models.EncoderDecoder.latentstore import LatentStore
from ainix_kernel.models.EncoderDecoder.nonretrieval import CopySpanPredictor, \
    get_copy_depth_discount


class RetrievalActionSelector(ActionSelector):
    def __init__(self, latent_store: LatentStore):
        super().__init__()
        self.latent_store = latent_store
        self.retrieve_dropout_p = 0.5
        self.max_query_retrieve_count_train = 50
        self.max_query_retrieve_count_infer = 10
        self.loss_func = torch.nn.MultiLabelSoftMarginLoss()
        self.span_predictor = CopySpanPredictor(latent_store.latent_size)

    def infer_predict(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        type_to_select: AInixType
    ) -> 'ActionResult':
        nearest_metadata, similarities, nearest_vals = self.latent_store.get_n_nearest_latents(
            latent_vec, type_to_select.name, max_results=self.max_query_retrieve_count_infer)
        # scale by (train_retrieve * dropout)/infer_retirev_count

    def forward_train(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        types_to_select: List[AInixType],
        expected: AstObjectChoiceSet,
        num_of_parents_with_copy_option: int
    ) -> torch.Tensor:
        assert len(types_to_select) == 1
        nearest_metadata, similarities, nearest_vals = self.latent_store.get_n_nearest_latents(
            latent_vec[0], types_to_select[0].name,
            max_results=self.max_query_retrieve_count_train)
        keep_mask = torch.rand(*nearest_metadata.impl_choices.shape) > self.retrieve_dropout_p
        similarities = similarities[keep_mask]
        impls_chosen = nearest_metadata.impl_choices[keep_mask]

        impl_scores, impl_keys = sparse_groupby_sum(similarities, impls_chosen)
        impls_indices_correct = are_indices_valid(impl_keys, self.ast_vocab, expected)
        # TODO weights
        loss = self.loss_func(impl_scores.unsqueeze(), impls_indices_correct.unsqueeze())
        if expected.copy_is_known_choice():
            span_pred_loss = self.span_predictor.train_predict_span(
                latent_vec, memory_tokens, types_to_select, expected)
            loss += get_copy_depth_discount(num_of_parents_with_copy_option) * span_pred_loss
