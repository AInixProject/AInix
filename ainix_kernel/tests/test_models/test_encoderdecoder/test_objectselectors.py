from collections import Counter
from typing import Tuple

import pytest

from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.model_util.vocab import BasicVocab
from ainix_kernel.models.EncoderDecoder.objectselector import *

SELECTORS = ["VectorizedSelector"]

def build_types() -> Tuple[TypeImplTensorMap, TypeContext]:
    tc = TypeContext()
    AInixType(tc, "Foo")
    AInixObject(tc, "oFoo1", "Foo")
    AInixObject(tc, "oFoo2", "Foo")
    AInixObject(tc, "oFoo3", "Foo")
    AInixObject(tc, "oFoo4", "Foo")
    AInixType(tc, "Zar")
    AInixObject(tc, "oZar1", "Zar")
    AInixObject(tc, "oZar2", "Zar")
    AInixType(tc, "Zaz")
    AInixObject(tc, "oZaz1", "Zaz")
    tc.finalize_data()
    return TypeImplTensorMap(tc), tc


class ToyVectorizer(VectorizerBase):
    def __init__(self, type_context: TypeContext):
        super().__init__()
        self.lookup = torch.zeros(type_context.get_object_count(), 3)
        self.lookup[0] = torch.Tensor([1, 1, 1])
        self.lookup[1] = torch.Tensor([-1, -1, -1])
        self.lookup[2] = torch.Tensor([3, 3, 3])
        self.lookup[3] = torch.Tensor([10, 0, 1])
        self.lookup[6] = torch.Tensor([3, 5, 1])

    def feature_len(self) -> int:
        return 3

    def forward(self, indicies) -> torch.Tensor:
        return self.lookup[indicies]


def build_selector(selector_name) -> Tuple[ObjectSelector, TypeContext]:
    if selector_name == "VectorizedSelector":
        type_m, tc = build_types()
        vectorizer = ToyVectorizer(tc)
        return VectorizedObjectSelector(type_m, vectorizer), tc


@pytest.mark.parametrize("selector_name", SELECTORS)
def test_query(selector_name):
    instance, type_context = build_selector(selector_name)
    impls, scores = instance(torch.Tensor([[1, 1, 1]]), [type_context.get_type_by_name("Foo")])
    # This is sort of fragile test as it explodes if change the toy inputs, but it works
    assert len(impls) == 1
    assert len(scores) == 1
    assert torch.all(scores[0] == torch.Tensor([3, -3, 9, 11]))
    impls, scores = instance(torch.Tensor([[1, 1, 1], [1, 0, -1]]),
                             [type_context.get_type_by_name("Foo"),
                              type_context.get_type_by_name("Zaz")])
    assert len(impls) == 2
    assert len(scores) == 2
    assert torch.all(scores[0] == torch.Tensor([3, -3, 9, 11]))
    assert torch.all(scores[1] == torch.Tensor([2]))
