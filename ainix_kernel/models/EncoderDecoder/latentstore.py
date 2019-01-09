"""Used to store the latent states of a model decoding for use in a latent
retrieval-based decoder."""
from typing import Dict, Tuple, List
import torch
import attr
import numpy as np

from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.typecontext import TypeContext, AInixObject, AInixType
from ainix_kernel.indexing.examplestore import ExamplesStore
import torch.nn.functional as F

COPY_IND = 10000


class LatentStore(torch.nn.Module):
    def __init__(self, type_context: TypeContext, latent_size: int, trainable_vals: bool = False):
        super().__init__()
        self.latent_size = latent_size
        # TODO (DNGros): Convert this to using vocab indexes rather than strings
        self.type_ind_to_latents: List[SingleTypeLatentStore] = \
            [SingleTypeLatentStore(latent_size, trainable_vals)
             for type in type_context.get_all_types()]
        self.is_in_write_mode = True
        self.allow_gradients = trainable_vals

    def add_latent(
        self,
        node: ObjectChoiceNode,
        latent: torch.Tensor,
        example_id: int,
        ydepth: int,
    ):
        ind = COPY_IND if node.copy_was_chosen else node.next_node_not_copy.implementation.ind
        latent_copy = latent.detach()
        latent_copy.requires_grad = self.allow_gradients
        self.type_ind_to_latents[node.type_to_choose.ind].add_latent(
            latent_copy, example_id, ydepth, ind)

    def get_n_nearest_latents(
        self,
        query_latent: torch.Tensor,
        typ: AInixType,
        max_results=50
    ) -> Tuple['LatentMetadataWrapper', torch.Tensor, torch.Tensor]:
        return self.type_ind_to_latents[typ.ind].get_n_nearest_latents(
            query_latent, max_results
        )

    def set_read(self):
        self.is_in_write_mode = False
        for s in self.type_ind_to_latents:
            s.set_read()
        if self.allow_gradients:
            self.type_ind_to_latents = torch.nn.ModuleList(self.type_ind_to_latents)

    #def parameters(self):
    #    if not self.allow_gradients:
    #        return []
    #    return (v.parameters() for v in self.type_ind_to_latents)


# TODO special value for copy
# TODO multilabel.
@attr.s(frozen=True, auto_attribs=True)
class LatentMetadataWrapper:
    """
    User friendly wrapper of for the latent metadata wrapper
    Data is a N by 3 array / tensor. N = number of latents stored
    # [N, 0] = the ids of the of the example it came from in the example_store
    # [N, 1] = when example [N, 0] y ast is viewed in the depth first manner, this
        represents the index into that dfs for the ObjectChoiceNode this came from
    # [N, 2] = the chosen implementation token id in the vocab.
    """
    data: torch.LongTensor

    @property
    def example_indxs(self) -> torch.Tensor:
        return self.data[:, 0]

    @property
    def y_inds(self) -> torch.Tensor:
        return self.data[:, 1]

    @property
    def impl_choices(self) -> torch.Tensor:
        return self.data[:, 2]


class SingleTypeLatentStore(torch.nn.Module):
    def __init__(self, latent_size: int, trainable_vals: bool = False):
        super().__init__()
        self.is_in_write_mode = True
        self.latents = []
        self.latent_metadatas = []  # For data format see LatentMetadataWrapper
        self.trainable_vals = trainable_vals

    def _get_similarities(self, query_latent: torch.Tensor) -> torch.Tensor:
        # Use dot product. Maybe should use cosine similarity?
        #print(f"LATENTSTORE {self.latents}")
        return torch.sum(query_latent*self.latents, dim=1)

    def add_latent(
        self,
        value: torch.Tensor,
        example_index: int,
        y_depth_idx: int,
        chosen_impl_idx: int
    ) -> None:
        if len(self.latent_metadatas) >= COPY_IND:
            raise ValueError("looks like yah need to think through scaleing")
        self.latents.append(value)
        self.latent_metadatas.append((example_index, y_depth_idx, chosen_impl_idx))

    def get_n_nearest_latents(
        self,
        query_latent: torch.Tensor,
        max_results: int
    ) -> Tuple[LatentMetadataWrapper, torch.Tensor, torch.Tensor]:
        """Queries to get the n nearest stored latent vectors to a latent vector

        Returns:
            Metadata object for the top n results sorted by similarity
            A tensor representing the similiarity score to each of those closes values.
            The actual vector values of those closest values.
        """
        if self.is_in_write_mode:
            raise ValueError("Must call set_read first")
        similarities = self._get_similarities(query_latent)
        similarities, sort_inds = similarities.sort(descending=True)
        take_count = max(similarities.shape[0], max_results)
        similarities, sort_inds = similarities[:take_count], sort_inds[:take_count]
        relevant_metadata = LatentMetadataWrapper(self.latent_metadatas[sort_inds])
        actual_vals: torch.Tensor = self.latents[:take_count]
        return relevant_metadata, similarities, actual_vals

    def set_read(self):
        self.is_in_write_mode = False
        self.latents = torch.stack(self.latents)
        if self.trainable_vals:
            self.latents = torch.nn.Parameter(self.latents)
        self.latent_metadatas = torch.LongTensor(self.latent_metadatas)

    #def parameters(self):
    #    if not self.trainable_vals:
    #        return None
    #    return self.latents


