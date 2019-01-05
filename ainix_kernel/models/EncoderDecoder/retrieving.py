from typing import List

import torch

from ainix_common.parsing.ast_components import AstObjectChoiceSet
from ainix_common.parsing.typecontext import AInixType
from ainix_kernel.models.EncoderDecoder.actionselector import ActionSelector
from ainix_kernel.models.EncoderDecoder.latentstore import LatentStore


class RetrievalActionSelector(ActionSelector):
    def __init__(self, latent_store: LatentStore):
        super().__init__()
        self.latent_store = latent_store
        self.max_query_retrieve_count_infer = 20
        self.max_query_retrieve_count_infer = 50

    def infer_predict(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        type_to_select: AInixType
    ) -> 'ActionResult':
        nearest_metadata, similarities, nearest_vals = self.latent_store.get_n_nearest_latents(
            latent_vec, type_to_select.name, max_results=self.max_query_retrieve_count_infer)

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
            max_results=self.max_query_retrieve_count_infer)
