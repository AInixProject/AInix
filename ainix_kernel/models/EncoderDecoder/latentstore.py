"""Used to store the latent states of a model decoding for use in a latent
retrieval-based decoder."""
from abc import abstractmethod, ABC
from typing import Dict, Tuple, List, Optional
import torch
import attr
import numpy as np

from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.typecontext import TypeContext, AInixObject, AInixType
from ainix_kernel.indexing.examplestore import ExamplesStore
import torch.nn.functional as F

COPY_IND = 10000

# Type -> tensor of latetnts
# Exampleid, dfs-depth/2 -> ind in the type tensor


@attr.s(frozen=True, auto_attribs=True)
class LatentDataWrapper:
    """
    User friendly wrapper of for the latent metadata wrapper
    Data is a N by L tensor. N = number of latents stored. L = the latent size.
    Metadata is a N by 3 array / tensor.
    # [0, N] = the ids of the of the example it came from in the example_store
    # [1, N] = when example [1, N] y ast is viewed in the depth first manner, this
        represents the index into that dfs for the ObjectChoiceNode this came from
    # [2, N] = the chosen implementation token id in the vocab.
    """
    latents: torch.Tensor
    metadata: torch.LongTensor

    @property
    def example_indxs(self) -> torch.Tensor:
        return self.metadata[0, :]

    @property
    def y_inds(self) -> torch.Tensor:
        return self.metadata[1, :]

    @property
    def impl_choices(self) -> torch.Tensor:
        return self.metadata[2, :]

    @classmethod
    def construct(
        cls,
        latents: torch.Tensor,
        example_indxs: torch.LongTensor,
        y_inds: torch.LongTensor,
        impl_choices: torch.LongTensor
    ) -> 'LatentDataWrapper':
        return cls(latents, torch.stack([example_indxs, y_inds, impl_choices]))


class LatentStore(ABC):
    @abstractmethod
    def get_n_nearest_latents(
        self,
        query_latent: torch.Tensor,
        type_ind: int,
        max_results=50
    ) -> Tuple[LatentDataWrapper, torch.Tensor]:
        pass

    @classmethod
    @abstractmethod
    def get_builder(cls, num_types: int, latent_size: int) -> 'LatentStoreBuilder':
        pass


class TorchLatentStore(LatentStore):
    def __init__(
        self,
        type_ind_to_latents: List[LatentDataWrapper],
        example_to_depths_to_ind_in_types: Optional[Dict[int, List[torch.LongTensor]]]
    ):
        self.type_ind_to_latents = type_ind_to_latents
        self.example_to_depths_to_ind_to_types = example_to_depths_to_ind_in_types

    def get_n_nearest_latents(
        self,
        query_latent: torch.Tensor,
        type_ind: int,
        max_results=50
    ) -> Tuple[LatentDataWrapper, torch.Tensor]:
        data = self.type_ind_to_latents[type_ind]
        similarities = self._get_similarities(query_latent, data.latents)
        similarities, sort_inds = similarities.sort(descending=True)
        take_count = max(similarities.shape[0], max_results)
        similarities, sort_inds = similarities[:take_count], sort_inds[:take_count]
        actual_vals: torch.Tensor = data.latents[:take_count]
        relevant_metadata = LatentDataWrapper(actual_vals, data.metadata[:, sort_inds])
        return relevant_metadata, similarities

    def _get_similarities(self, query_latent: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Use dot product. Maybe should use cosine similarity?
        return torch.sum(query_latent*values, dim=1)

    @classmethod
    def get_builder(cls, num_types: int, latent_size: int) -> 'LatentStoreBuilder':
        return TorchLatentStoreBuilder(num_types, latent_size)


class LatentStoreBuilder(ABC):
    pass


class TorchLatentStoreBuilder(LatentStoreBuilder):
    def __init__(self, num_types: int, latent_size: int):
        self.type_ind_to_example_id: List[List[int]] = [[] for _ in range(num_types)]
        self.type_ind_to_impl_choice: List[List[int]] = [[] for _ in range(num_types)]
        self.type_ind_to_y_dfsdeth: List[List[int]] = [[] for _ in range(num_types)]
        self.example_id_to_inds_in_type: Dict[int, List[int]] = {}
        self.latent_size = latent_size

    def add_example(
        self,
        example_id: int,
        ast: ObjectChoiceNode
    ):
        ind_map = []
        for y_depth, pointer in enumerate(ast.depth_first_iter()):
            if isinstance(pointer.cur_node, ObjectChoiceNode):
                assert y_depth % 2 == 0
                type_ind = pointer.cur_node.type_to_choose.ind
                ind_map.append(len(self.type_ind_to_example_id[type_ind]))
                self.type_ind_to_example_id[type_ind].append(example_id)
                chosen_impl_id = COPY_IND if pointer.cur_node.copy_was_chosen \
                    else pointer.cur_node.next_node_not_copy.implementation.ind
                self.type_ind_to_impl_choice[type_ind].append(chosen_impl_id)
                self.type_ind_to_y_dfsdeth[type_ind].append(y_depth)
        assert example_id not in self.example_id_to_inds_in_type
        self.example_id_to_inds_in_type[example_id] = ind_map

    def produce_result(self) -> TorchLatentStore:
        return TorchLatentStore(
            type_ind_to_latents=[
                LatentDataWrapper.construct(
                    latents=torch.randn(len(tex), self.latent_size),
                    example_indxs=torch.LongTensor(tex),
                    y_inds=torch.LongTensor(tdfs),
                    impl_choices=torch.LongTensor(timpl)
                ) for tex, tdfs, timpl in zip(
                    self.type_ind_to_example_id,
                    self.type_ind_to_y_dfsdeth,
                    self.type_ind_to_impl_choice
                )
            ],
            example_to_depths_to_ind_in_types={
                ex_id: torch.LongTensor(vs)
                for ex_id, vs in self.example_id_to_inds_in_type.items()
            }
        )


#class LatentStore:#torch.nn.Module):
#    def __init__(self, type_context: TypeContext, latent_size: int, trainable_vals: bool = False):
#        super().__init__()
#        self.latent_size = latent_size
#        # TODO (DNGros): Convert this to using vocab indexes rather than strings
#        self.type_ind_to_latents: List[SingleTypeLatentStore] = \
#            [SingleTypeLatentStore(latent_size, trainable_vals)
#             for type in type_context.get_all_types()]
#        self.is_in_write_mode = True
#        self.allow_gradients = trainable_vals
#
#    def add_latent(
#        self,
#        node: ObjectChoiceNode,
#        latent: torch.Tensor,
#        example_id: int,
#        ydepth: int,
#    ):
#        ind = COPY_IND if node.copy_was_chosen else node.next_node_not_copy.implementation.ind
#        latent_copy = latent.detach()
#        latent_copy.requires_grad = self.allow_gradients
#        self.type_ind_to_latents[node.type_to_choose.ind].add_latent(
#            latent_copy, example_id, ydepth, ind)
#
#    def get_n_nearest_latents(
#        self,
#        query_latent: torch.Tensor,
#        typ: AInixType,
#        max_results=50
#    ) -> Tuple['LatentMetadataWrapper', torch.Tensor, torch.Tensor]:
#        return self.type_ind_to_latents[typ.ind].get_n_nearest_latents(
#            query_latent, max_results
#        )
#
#    def set_read(self):
#        self.is_in_write_mode = False
#        for s in self.type_ind_to_latents:
#            s.set_read()
#        if self.allow_gradients:
#            self.type_ind_to_latents = torch.nn.ModuleList(self.type_ind_to_latents)
#
#    #def parameters(self):
#    #    if not self.allow_gradients:
#    #        return []
#    #    return (v.parameters() for v in self.type_ind_to_latents)
#


#
#
#class SingleTypeLatentStore:#torch.nn.Module):
#    def __init__(self, latent_size: int, trainable_vals: bool = False):
#        super().__init__()
#        self.is_in_write_mode = True
#        self.latents = []
#        self.latent_metadatas = []  # For data format see LatentMetadataWrapper
#        self.trainable_vals = trainable_vals
#
#    def _get_similarities(self, query_latent: torch.Tensor) -> torch.Tensor:
#        # Use dot product. Maybe should use cosine similarity?
#        #print(f"LATENTSTORE {self.latents}")
#        return torch.sum(query_latent*self.latents, dim=1)
#
#    def add_latent(
#        self,
#        value: torch.Tensor,
#        example_index: int,
#        y_depth_idx: int,
#        chosen_impl_idx: int
#    ) -> None:
#        if len(self.latent_metadatas) >= COPY_IND:
#            raise ValueError("looks like yah need to think through scaleing")
#        self.latents.append(value)
#        self.latent_metadatas.append((example_index, y_depth_idx, chosen_impl_idx))
#
#    def get_n_nearest_latents(
#        self,
#        query_latent: torch.Tensor,
#        max_results: int
#    ) -> Tuple[LatentMetadataWrapper, torch.Tensor, torch.Tensor]:
#        """Queries to get the n nearest stored latent vectors to a latent vector
#
#        Returns:
#            Metadata object for the top n results sorted by similarity
#            A tensor representing the similiarity score to each of those closes values.
#            The actual vector values of those closest values.
#        """
#        if self.is_in_write_mode:
#            raise ValueError("Must call set_read first")
#        similarities = self._get_similarities(query_latent)
#        similarities, sort_inds = similarities.sort(descending=True)
#        take_count = max(similarities.shape[0], max_results)
#        similarities, sort_inds = similarities[:take_count], sort_inds[:take_count]
#        relevant_metadata = LatentMetadataWrapper(self.latent_metadatas[sort_inds])
#        actual_vals: torch.Tensor = self.latents[:take_count]
#        return relevant_metadata, similarities, actual_vals
#
#    def set_read(self):
#        self.is_in_write_mode = False
#        self.latents = torch.stack(self.latents)
#        if self.trainable_vals:
#            self.latents = torch.nn.Parameter(self.latents)
#        self.latent_metadatas = torch.LongTensor(self.latent_metadatas)
#
#    #def parameters(self):
#    #    if not self.trainable_vals:
#    #        return None
#    #    return self.latents


def make_latent_store_from_examples_and_model(
    model,
    examples: ExamplesStore,
    replacers
) -> LatentStore:
    for example in examples.get_all_examples():
        pass
