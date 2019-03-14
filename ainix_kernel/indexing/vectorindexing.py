"""Generic stuff for storing a relation between a vector and a python object"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Dict
import numpy as np
import attr
import torch
import pandas as pd
import torch.nn.functional as F


class SimilarityMeasure(Enum):
    DOT = "DOT"
    COS = "COS"


class VectorDatabase(ABC):
    """An abc for a database for something that lets you query nearest neighbors
    of some vectors and get associated values"""
    @abstractmethod
    def get_n_nearest(
        self,
        query: np.ndarray,
        similarity_metric: 'SimilarityMeasure' = SimilarityMeasure.DOT,
        max_results=50
    ) -> Tuple[List[Tuple], torch.Tensor]:
        pass

    def get_nearest(
        self,
        query: np.ndarray,
        similarity_metric: 'SimilarityMeasure' = SimilarityMeasure.DOT
    ) -> Tuple[Tuple, torch.Tensor]:
        values, similarites = self.get_n_nearest(query, similarity_metric, 1)
        return values[0], similarites[0]


class VectorDatabaseBuilder(ABC):
    """Potentially used for immutable vector databases"""
    def __init__(
        self,
        key_dimensionality: int,
        value_fields: List['VDBField']
    ):
        self.key_dims = key_dimensionality
        self.value_fields = value_fields
        self.value_df = pd.DataFrame()

    @abstractmethod
    def add_data(
        self,
        key: np.ndarray,
        data: Tuple
    ):
        pass

    @abstractmethod
    def produce_result(self) -> VectorDatabase:
        pass


class TorchVectorDatabase(VectorDatabase):
    """A implementation of VectorDatabase just based off of big torch vectors.
    This should work fine enough for now. In the future we might need to implement
    a database based off some fancy 3rd party knn search function"""
    def __init__(
        self,
        keys_tensor: torch.Tensor,
        values_df:  pd.DataFrame,
        value_fields: List['VDBField']
    ):
        self._keys = keys_tensor
        self._values = values_df
        self._value_fields = value_fields
        self._value_names = [vf.name for vf in value_fields]

    def get_n_nearest(
        self,
        query: torch.Tensor,
        similarity_metric: 'SimilarityMeasure' = SimilarityMeasure.DOT,
        max_results=50
    ) -> Tuple[List[Tuple], torch.Tensor]:
        if similarity_metric == SimilarityMeasure.DOT:
            similarities = self._dot_similarity(query, self._keys)
        elif similarity_metric == SimilarityMeasure.COS:
            similarities = self._cos_similarity(query, self._keys)
        else:
            raise ValueError(f"Unsupported similarity {similarity_metric}")
        similarities, sort_inds = similarities.sort(descending=True)
        take_count = min(similarities.shape[0], max_results)
        similarities, sort_inds = similarities[:take_count], sort_inds[:take_count]
        values = [tuple(x) for x in self._values.iloc[sort_inds.data.numpy()].values]
        return values, similarities

    def _dot_similarity(self, query_latent: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Use dot product. Maybe should use cosine similarity?
        return torch.sum(query_latent*values, dim=1)

    def _cos_similarity(self, query_latent: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Use dot product. Maybe should use cosine similarity?
        return F.cosine_similarity(query_latent, values, dim=1)


class TorchVectorDatabaseBuilder(VectorDatabaseBuilder):
    def __init__(self, key_dimensionality: int, value_fields: List['VDBField']):
        super().__init__(key_dimensionality, value_fields)
        self.key_dims = key_dimensionality
        self.keys = []
        self.datas = []
        self.value_fields = value_fields
        self.already_produced = False

    def add_data(
        self,
        key: np.ndarray,
        data: Tuple
    ):
        if self.already_produced:
            raise ValueError("Builder cannot be reused")
        if len(key.shape) != 1:
            raise ValueError("expected vector as key")
        if len(key) != self.key_dims:
            raise ValueError("key does not match expected dimensionality")
        if len(data) != len(self.value_fields):
            raise ValueError("unexpected number of fields")
        self.keys.append(key)
        self.datas.append(data)

    def produce_result(self) -> TorchVectorDatabase:
        dtypes = [FIELD_TYPE_TO_D_TYPE[type(vf)] for vf in self.value_fields]
        out = TorchVectorDatabase(
            keys_tensor=torch.tensor(self.keys).float(),
            values_df=pd.DataFrame(
                self.datas,
                columns=[vf.name for vf in self.value_fields]
            ),
            value_fields=self.value_fields
        )
        # Free memory the builder is useing
        self.keys = None
        self.datas = None
        return out


class VDBField(ABC):
    name: str


@attr.s(auto_attribs=True, frozen=True)
class VDBIntField(VDBField):
    name: str


@attr.s(auto_attribs=True, frozen=True)
class VDBStringField(VDBField):
    name: str


@attr.s(auto_attribs=True, frozen=True)
class VDBObjectField(VDBField):
    name: str


FIELD_TYPE_TO_D_TYPE = {
    VDBIntField: 'i',
    VDBStringField: 'U',
    VDBObjectField: 'O'
}


def df_empty(columns, dtypes, index=None):
    """From
    https://stackoverflow.com/questions/36462257/
        create-empty-dataframe-in-pandas-specifying-column-types

    by user48956"""
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

