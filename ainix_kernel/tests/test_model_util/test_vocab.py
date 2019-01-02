from collections import Counter
from typing import Tuple

from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.model_util.vocab import *
import pytest

@pytest.fixture()
def toy_vocab_and_context() -> Tuple[Vocab, TypeContext]:
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
    return ast_vocab, tc

@pytest.fixture()
def string_vocab() -> Vocab:
    builder = CounterVocabBuilder(specials=[])
    builder.add_sequence(["foo", "bar", "baz"])
    builder.add_sequence(["foo", "moo"])
    return builder.produce_vocab()


def test_vocab(string_vocab):
    assert len(string_vocab) == 4
    assert string_vocab.token_to_index("foo") == 0
    assert string_vocab.index_to_token(0) == "foo"


def test_vocab_token_seq_to_inds(string_vocab):
    assert torch.all(string_vocab.token_seq_to_indices(["foo", "baz"]) ==
                     torch.LongTensor([0, 2]))
    assert torch.all(string_vocab.token_seq_to_indices([["foo", "baz"], ["bar", "moo"]]) ==
                     torch.LongTensor([[0, 2], [1, 3]]))


def test_vocab_seq_to_tok(string_vocab):
    res = string_vocab.torch_indices_to_tokens(torch.LongTensor([[0, 2], [1, 3]]))
    assert tuple(res[0]) == ("foo", "baz")
    assert tuple(res[1]) == ("bar", "moo")


def test_ast_valid_thing(toy_vocab_and_context):
    toy_vocab, type_context = toy_vocab_and_context
    foo_type = type_context.get_type_by_name("Foo")
    f1 = type_context.get_object_by_name("oFoo1")
    f2 = type_context.get_object_by_name("oFoo2")
    f3 = type_context.get_object_by_name("oFoo3")
    f4 = type_context.get_object_by_name("oFoo4")
    ast_set = AstObjectChoiceSet(foo_type, None)
    ast_set.is_known_choice = lambda n: True if n in ("oFoo1", "oFoo3") else False
    to_test = toy_vocab.token_seq_to_indices(
        [[f1, f2, f3]], True)
    result = are_indices_valid(to_test, toy_vocab, ast_set)
    assert torch.all(torch.Tensor([[1, 0, 1]]) == result)


def test_counter_vocab_serialize(string_vocab):
    save = string_vocab.get_save_state_dict()
    new = CounterVocab.create_from_save_state_dict(save)
    print(vars(string_vocab))
    print(vars(new))
    assert list(string_vocab.itos) == list(new.itos)
    assert string_vocab.stoi == new.stoi

