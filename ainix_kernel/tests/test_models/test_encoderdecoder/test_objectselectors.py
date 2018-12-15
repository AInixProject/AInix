from collections import Counter
from typing import Tuple

import pytest

from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.model_util.vocab import CounterVocab
from ainix_kernel.models.EncoderDecoder.objectselector import *

SELECTORS = ["VectorizedSelector"]

def build_types() -> Tuple[TypeImplTensorMap, Vocab, TypeContext]:
    tc = TypeContext()
    counter = Counter()
    counter[AInixType(tc, "Foo")] += 1
    counter[AInixObject(tc, "oFoo1", "Foo")] += 1
    counter[AInixObject(tc, "oFoo2", "Foo")] += 1
    counter[AInixObject(tc, "oFoo3", "Foo")] += 1
    counter[AInixObject(tc, "oFoo4", "Foo")] += 1
    counter[AInixType(tc, "Bar")] += 1
    counter[AInixObject(tc, "oBar1", "Bar")] += 1
    counter[AInixObject(tc, "oBar2", "Bar")] += 1
    counter[AInixType(tc, "Baz")] += 1
    counter[AInixObject(tc, "oBaz1", "Baz")] += 1
    ast_vocab = CounterVocab(counter, specials=[])
    return TypeImplTensorMap(ast_vocab), ast_vocab, tc


class ToyVectorizer(VectorizerBase):
    def __init__(self, vocab):
        super().__init__()
        self.lookup = torch.zeros(len(vocab), 3)
        self.lookup[6] = torch.Tensor([1, 1, 1])
        self.lookup[7] = torch.Tensor([-1, -1, -1])
        self.lookup[8] = torch.Tensor([3, 3, 3])
        self.lookup[5] = torch.Tensor([10, 0, 1])

    def feature_len(self) -> int:
        return 3

    def forward(self, indicies) -> torch.Tensor:
        return self.lookup[indicies]


def build_selector(selector_name) -> Tuple[ObjectSelector, TypeContext]:
    if selector_name == "VectorizedSelector":
        type_m, vocab, tc = build_types()
        vectorizer = ToyVectorizer(vocab)
        return VectorizedObjectSelector(type_m, vectorizer), tc


@pytest.mark.parametrize("selector_name", SELECTORS)
def test_query(selector_name):
    instance, type_context = build_selector(selector_name)
    impls, scores = instance(torch.Tensor([[1, 1, 1]]), [type_context.get_type_by_name("Foo")])
    # This is sort of fragile test as it explodes if change the toy inputs, but it works
    assert len(impls) == 1
    assert len(scores) == 1
    assert torch.all(scores[0] == torch.Tensor([3, -3, 9, 0]))
    impls, scores = instance(torch.Tensor([[1, 1, 1], [1, 0, -1]]),
                             [type_context.get_type_by_name("Foo"),
                              type_context.get_type_by_name("Baz")])
    assert len(impls) == 2
    assert len(scores) == 2
    assert torch.all(scores[0] == torch.Tensor([3, -3, 9, 0]))
    assert torch.all(scores[1] == torch.Tensor([9]))
