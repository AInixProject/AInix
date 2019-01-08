import itertools

from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.typecontext import AInixObject
from ainix_kernel.models.EncoderDecoder.latentstore import LatentStore
from ainix_kernel.models.EncoderDecoder.nonretrieval import *
import pytest

from ainix_kernel.models.EncoderDecoder.retrieving import RetrievalActionSelector


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

    store = LatentStore(tc, 3)
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
        torch.Tensor([0, 0, 3]), 2, 0
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


def test_train_retriever_selector(latent_store_stuff):
    latent_store, tc, valid_choices = latent_store_stuff
    inputs = [(torch.LongTensor([0]), c) for i, c in enumerate(valid_choices)]
    embed = torch.nn.Embedding(len(valid_choices), 3)
    instance = RetrievalActionSelector(latent_store, tc)
    params = itertools.chain(instance.parameters(), embed.parameters())
    print(params)
    optim = torch.optim.Adam(params)

    def do_train():
        optim.zero_grad()
        loss = 0
        for x, y in inputs:
            loss += instance.forward_train(embed(x), None, [tc.get_type_by_name("FT")], y, 0)
        loss.backward()
        optim.step()
        return loss

    for e in range(100):
        print(do_train())
