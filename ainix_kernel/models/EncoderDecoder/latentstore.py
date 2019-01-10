"""Used to store the latent states of a model decoding for use in a latent
retrieval-based decoder."""
import math
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

    def get_default_trainer(self) -> 'LatentStoreTrainer':
        raise ValueError("This latent store does not support training")


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

    def get_default_trainer(self) -> 'LatentStoreTrainer':
        return TorchStoreTrainerAdam(self)

    def get_latent_for_example(
        self,
        type_id: int,
        example_id: int,
        dfs_depth: int
    ) -> Optional[torch.Tensor]:
        if example_id not in self.example_to_depths_to_ind_to_types:
            return None
        depth_to_inds_in_type = self.example_to_depths_to_ind_to_types[example_id]
        if int(dfs_depth / 2) >= len(depth_to_inds_in_type):
            return None
        return self.type_ind_to_latents[type_id].latents[depth_to_inds_in_type[int(dfs_depth / 2)]]


class LatentStoreBuilder(ABC):
    @abstractmethod
    def add_example(
        self,
        example_id: int,
        ast: ObjectChoiceNode
    ):
        pass

    @abstractmethod
    def produce_result(self) -> LatentStore:
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



class LatentStoreTrainer(ABC):
    """A wrapper around a latent store trainer which sorta acts equivolently
    to torch.nn.optim. When you make queries on the trainer, it will update the
    stored value for the example you are now retrieving for."""
    @abstractmethod
    def update_value(
        self,
        type_ind: int,
        example_ind: int,
        dfs_depth: int,
        new_latent: torch.Tensor
    ) -> torch.Tensor:
        """Update a stored latent. Return the actual new value"""
        pass


class TorchStoreTrainerAdam(LatentStoreTrainer):
    def __init__(
        self,
        store_to_train: TorchLatentStore,
        lr=0.2e-3,
        beta1=0.87,
        beta2=0.9992,
        eps=1e-8
    ):
        self.store = store_to_train
        # We currently use Adam optimization to update values
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        # Moments are stored as dims (T, I, 2, V)
        # T = number of types.
        # I = the number of examples of that type
        # [t, i, 0, V] = first moment. So exp avg of gradient
        # [t, i, 1, V] = second moment. So exp avg of square gradient
        # V = the size of latents. So like latents might be 64 len vectors
        self.type_ind_to_moments = [
            torch.zeros((data.latents.shape[0], 2, data.latents.shape[1]))
            for data in self.store.type_ind_to_latents
        ]
        self.step_counts = [
            torch.zeros(data.latents.shape[0])
            for data in self.store.type_ind_to_latents
        ]
        # There may be a way to just make the LatentStore a nn.Module which can
        # then return its parameters to be optimized in the normal optim step.
        # However, this has several issues. We need to mask out the gradients
        # for non-queried latents. Also we can't mask with 0, because that will
        # mess up the moment estimates. There could be a way to solve this, but
        # for now we'll just reimplement our own optimzer. This has advantages
        # anyways in being more flexible if we eventually move to a on-disk
        # latentstore, plus it potenitally stores less gradient vals.

    def update_value(
        self,
        type_ind: int,
        example_ind: int,
        dfs_depth: int,
        new_latent: torch.Tensor
    ) -> torch.Tensor:
        """Do an Adam update of a value in the latent store"""
        ind_in_type = self.store.example_to_depths_to_ind_to_types[example_ind][int(dfs_depth / 2)]
        d = self.store.type_ind_to_latents[type_ind]
        assert d.example_indxs[ind_in_type] == example_ind
        assert d.y_inds[ind_in_type] == dfs_depth
        current_latent = d.latents[ind_in_type]
        error = new_latent - current_latent
        # Code ref https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
        self.type_ind_to_moments[type_ind][ind_in_type, 0].mul_(
            self.beta1).add_(1 - self.beta1, error)
        self.type_ind_to_moments[type_ind][ind_in_type, 1].mul_(self.beta2).addcmul_(
            1 - self.beta2, error, error)
        denom = self.type_ind_to_moments[type_ind][ind_in_type, 1].sqrt().add_(self.eps)
        self.step_counts[type_ind][ind_in_type] += 1
        step = int(self.step_counts[type_ind][ind_in_type])
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        step_size = self.lr * math.sqrt(bias_correction1) / bias_correction2
        # not this is equivolent so far to pytorch code, except here we use postive step size
        d.latents[ind_in_type].addcdiv_(
            step_size, self.type_ind_to_moments[type_ind][ind_in_type, 0], denom)
        return d.latents[ind_in_type]
    pass
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