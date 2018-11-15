from collections import defaultdict
from typing import List

import torch
from torch import nn
from abc import ABC
from ainix_common.parsing.typecontext import AInixType, AInixObject
from ainix_kernel.model_util.vectorizers import VectorizerBase
from ainix_kernel.model_util.vocab import Vocab


class TypeImplTensorMap:
    """Maps objects to tensors"""
    def __init__(self, ast_vocab: Vocab):
        self.vocab_size = len(ast_vocab)
        self._type_to_impl_tensor = defaultdict(list)
        # Pick out all the objects
        for ind, el in ast_vocab.items():
            if isinstance(el, AInixObject):
                self._type_to_impl_tensor[el.type].append(ind)
        # convert to torch tensors
        self._type_to_impl_tensor = {type_: torch.LongTensor(v)
                                     for type_, v in self._type_to_impl_tensor.items()}

    def get_tensor_of_implementations(self, types: List[AInixType]):
        return torch.stack([self._type_to_impl_tensor[t] for t in types])


class ObjectSelector(nn.Module, ABC):
    """Converts between a vector and scoring on a set of objects"""
    pass

class VectorizedObjectSelector(nn.Module, ABC):
    def __init__(self, type_tensor_map: TypeImplTensorMap, ast_vectorizer: VectorizerBase):
        super().__init__()
        self.type_tensor_map = type_tensor_map
        self.ast_vectorizer = ast_vectorizer

    def forward(self, vectors: torch.Tensor, types_to_choose: List[AInixType]):
        """
        Scores the most likely implementations of certain type.
        Args:
            vectors: Tensor of shape (batch, features_to_select_on)
            types_to_choose: A list of of what type to select for each batch

        Returns:

        """
        impls = self.type_tensor_map.get_tensor_of_implementations(types_to_choose)
        out_scores = []
        # TODO (DNGros): figure out how to batch. Maybe use the pack_pad magic. Or group on type
        for impl_set, q_vector in zip(impls, vectors):
            impl_vectors = self.ast_vectorizer(impl_set)
            # Here we use dot product as similarity. Not sure if this is a
            # good idea or not. Cosine dist might make more sense.
            similarity = torch.sum(q_vector*impl_vectors, dim=1)
            out_scores.append(similarity)
        return impls, out_scores


def get_default_object_selector(ast_vocab: Vocab, ast_vectorizer: VectorizerBase):
    type_tensor_map = TypeImplTensorMap(ast_vocab)
    return VectorizedObjectSelector(type_tensor_map, ast_vectorizer)