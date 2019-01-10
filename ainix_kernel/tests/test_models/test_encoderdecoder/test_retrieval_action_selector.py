import itertools

from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.typecontext import AInixObject
from ainix_kernel.models.EncoderDecoder.latentstore import LatentStore, TorchLatentStore
from ainix_kernel.models.EncoderDecoder.nonretrieval import *
import pytest

from ainix_kernel.models.EncoderDecoder.retrieving import RetrievalActionSelector
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq


@pytest.fixture()
def latent_store_stuff() -> Tuple[LatentStore, TypeContext, List[AstObjectChoiceSet]]:
    torch.manual_seed(1)
    tc = TypeContext()
    ft = AInixType(tc, "FT")
    fo1 = AInixObject(tc, "FO1", "FT")
    fo2 = AInixObject(tc, "FO2", "FT")
    fo3 = AInixObject(tc, "FO3", "FT")
    bt = AInixType(tc, "BT")
    bo1 = AInixObject(tc, "BO1", "BT")
    AInixObject(tc, "BO2", "BT")
    tc.finalize_data()

    builder = TorchLatentStore.get_builder(tc.get_type_count(), 3)
    valid_choices = []
    oc = ObjectChoiceNode(ft, ObjectNode(fo1))
    builder.add_example(0, oc)
    s1 = AstObjectChoiceSet(ft)
    s1.add(oc, True, 1, 1)
    valid_choices.append((0, 0, s1))

    oc = ObjectChoiceNode(ft, ObjectNode(fo2))
    builder.add_example(1, oc)
    s1 = AstObjectChoiceSet(ft)
    s1.add(oc, True, 1, 1)
    valid_choices.append((1, 0, s1))

    builder.add_example(2, ObjectChoiceNode(ft, ObjectNode(fo3)))
    return builder.produce_result(), tc, valid_choices


#def test_train_retriever_selector_no_train(latent_store_stuff):
#    latent_store, tc, valid_choices = latent_store_stuff
#    instance = RetrievalActionSelector(latent_store, tc, retrieve_dropout_p=0)
#    pred = instance.infer_predict(torch.Tensor([1, 2, 3]), None, tc.get_type_by_name("FT"))
#    assert isinstance(pred, ProduceObjectAction)
#    assert pred.implementation.name == "FO1"
#    pred = instance.infer_predict(torch.Tensor([-2, 0, 3]), None, tc.get_type_by_name("FT"))
#    assert isinstance(pred, ProduceObjectAction)
#    assert pred.implementation.name == "FO2"
#    pred = instance.infer_predict(torch.Tensor([3, 0, -3]), None, tc.get_type_by_name("FT"))
#    assert isinstance(pred, ProduceObjectAction)
#    assert pred.implementation.name == "FO3"


def test_train_retriever_selector(latent_store_stuff):
    latent_store, tc, valid_choices = latent_store_stuff
    latent_size = 3
    inputs = [(torch.randn(latent_size), c) for i, c in enumerate(valid_choices)]
    instance = RetrievalActionSelector(latent_store, tc, retrieve_dropout_p=0)
    instance.start_train_session()
    optim = torch.optim.Adam(instance.parameters(), lr=1e-2)
    print(inputs)

    def do_train():
        optim.zero_grad()
        loss = 0
        for x, (example_id, step, astset) in inputs:
            loss += instance.forward_train(x.unsqueeze(0), None, [tc.get_type_by_name("FT")],
                                           astset, 0, [example_id], [step])
        try:
            loss.backward()
        except RuntimeError as e:
            if "element 0 of tensors does not require" not in str(e):
                raise e
        optim.step()
        return loss

    for e in range(50):
        loss = do_train()
        #print("LOSS", loss)
        #s: TorchLatentStore = instance.latent_store
        #print("LATENTS", s.type_ind_to_latents)

    for x, (example_id, step, astset) in inputs:
        pred = instance.infer_predict(x, None, tc.get_type_by_name("FT"))
        #print("x", x)
        #print("pred", pred)
        assert isinstance(pred, ProduceObjectAction)
        assert astset.is_known_choice(pred.implementation.name)
