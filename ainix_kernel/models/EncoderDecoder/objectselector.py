# TODO (DNGros): after refactoring to ActionSelectors the name of this is confusing.
from collections import defaultdict
from typing import List

import torch
from torch import nn
from abc import ABC
from ainix_common.parsing.typecontext import AInixType, AInixObject, TypeContext
from ainix_kernel.model_util.vectorizers import VectorizerBase
from ainix_kernel.model_util.vocab import Vocab


class TypeImplTensorMap:
    """Maps types to long tensors that contains the index ids for all the
    implementations of that type"""
    def __init__(self, type_context: TypeContext):
        self._type_to_impl_tensor = [None] * type_context.get_type_count()
        for typ in type_context.get_all_types():
            self._type_to_impl_tensor[typ.ind] = \
                torch.LongTensor([impl.ind for impl in type_context.get_implementations(typ)])

    def get_tensor_of_implementations(self, types: List[AInixType]):
        return [self._type_to_impl_tensor[t.ind] for t in types]


class ObjectSelector(nn.Module, ABC):
    """Converts between a vector and scoring on a set of objects"""
    pass


class VectorizedObjectSelector(nn.Module, ABC):
    def __init__(self, type_tensor_map: TypeImplTensorMap, object_vectorizer: VectorizerBase):
        super().__init__()
        self.type_tensor_map = type_tensor_map
        self.object_vectorizer = object_vectorizer

    def forward(self, vectors: torch.Tensor, types_to_choose: List[AInixType]):
        """
        Scores the most likely implementations of certain type.
        Args:
            vectors: Tensor of shape (batch, features_to_select_on)
            types_to_choose: A list of of what type to select for each item in batch

        Returns:

        """
        impls = self.type_tensor_map.get_tensor_of_implementations(types_to_choose)
        out_scores = []
        # TODO (DNGros): figure out how to batch. Maybe use the pack_pad magic. Or group on type
        for impl_set, q_vector in zip(impls, vectors):
            impl_vectors = self.object_vectorizer(impl_set)
            # Here we use dot product as similarity. Not sure if this is a
            # good idea or not. Cosine dist might make more sense.
            similarity = torch.sum(q_vector*impl_vectors, dim=1)
            out_scores.append(similarity)
        return impls, out_scores


def get_default_object_selector(type_context: TypeContext, object_vectorizer: VectorizerBase):
    type_tensor_map = TypeImplTensorMap(type_context)
    return VectorizedObjectSelector(type_tensor_map, object_vectorizer)
