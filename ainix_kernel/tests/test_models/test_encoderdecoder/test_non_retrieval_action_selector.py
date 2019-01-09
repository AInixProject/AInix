import itertools

from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.typecontext import AInixObject
from ainix_kernel.models.EncoderDecoder.latentstore import LatentStore
from ainix_kernel.models.EncoderDecoder.nonretrieval import *
import pytest

from ainix_kernel.models.EncoderDecoder.retrieving import RetrievalActionSelector
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq


@pytest.fixture()
def latent_store_stuff() -> Tuple[LatentStore, TypeContext, List[AstObjectChoiceSet]]:
    tc = TypeContext()
    ft = AInixType(tc, "FT")
    fo1 = AInixObject(tc, "FO1", "FT")
    fo2 = AInixObject(tc, "FO2", "FT")
    fo3 = AInixObject(tc, "FO3", "FT")
    bt = AInixType(tc, "BT")
    bo1 = AInixObject(tc, "BO1", "BT")
    AInixObject(tc, "BO2", "BT")
    tc.finalize_data()

    store = LatentStore(tc, 3, False)
    valid_choices = []
    oc = ObjectChoiceNode(ft, ObjectNode(fo1))
    store.add_latent(
        oc,
        torch.Tensor([1, 2, 3]),
        0, 0
    )
    s1 = AstObjectChoiceSet(ft)
    s1.add(oc, True, 1, 1)
    valid_choices.append(s1)

    oc = ObjectChoiceNode(ft, ObjectNode(fo2))
    store.add_latent(
        oc,
        torch.Tensor([-2, 0, 3]), 2, 0
    )
    s1 = AstObjectChoiceSet(ft)
    s1.add(oc, True, 1, 1)
    valid_choices.append(s1)

    store.add_latent(
        ObjectChoiceNode(ft, ObjectNode(fo3)),
        torch.Tensor([3, 0, -3]), 1, 3
    )
    store.add_latent(
        ObjectChoiceNode(bt, ObjectNode(bo1)),
        torch.Tensor([1, 3, -1]), 1, 3
    )
    store.set_read()
    return store, tc, valid_choices


def test_train_retriever_selector_no_train(latent_store_stuff):
    latent_store, tc, valid_choices = latent_store_stuff
    instance = RetrievalActionSelector(latent_store, tc, retrieve_dropout_p=0)
    pred = instance.infer_predict(torch.Tensor([1, 2, 3]), None, tc.get_type_by_name("FT"))
    assert isinstance(pred, ProduceObjectAction)
    assert pred.implementation.name == "FO1"
    pred = instance.infer_predict(torch.Tensor([-2, 0, 3]), None, tc.get_type_by_name("FT"))
    assert isinstance(pred, ProduceObjectAction)
    assert pred.implementation.name == "FO2"
    pred = instance.infer_predict(torch.Tensor([3, 0, -3]), None, tc.get_type_by_name("FT"))
    assert isinstance(pred, ProduceObjectAction)
    assert pred.implementation.name == "FO3"


def test_train_retriever_selector(latent_store_stuff):
    latent_store, tc, valid_choices = latent_store_stuff
    inputs = [(torch.LongTensor([i]), c) for i, c in enumerate(valid_choices)]
    torch.manual_seed(1)
    embed = torch.nn.Embedding(len(valid_choices), 3)
    instance = RetrievalActionSelector(latent_store, tc, retrieve_dropout_p=0)
    params = itertools.chain(instance.parameters(), embed.parameters())
    print(params)
    optim = torch.optim.Adam(params, lr=1e-2)

    def do_train():
        optim.zero_grad()
        loss = 0
        for x, y in inputs:
            loss += instance.forward_train(embed(x), None, [tc.get_type_by_name("FT")], y, 0)
        loss.backward()
        optim.step()
        return loss

    for e in range(800):
        do_train()
        #print("LOSS", do_train())
        #print("EMBED", embed.weight)

    for x, y in inputs:
        x_v = embed(x)
        pred = instance.infer_predict(x_v, None, tc.get_type_by_name("FT"))
        #print("X_V", x_v)
        #print("pred", pred)
        assert isinstance(pred, ProduceObjectAction)
        assert y.is_known_choice(pred.implementation.name)
