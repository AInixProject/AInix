import pytest

from ainix_common.parsing.ast_components import ObjectNode
from ainix_common.parsing.typecontext import AInixArgument
from ainix_kernel.models.EncoderDecoder.latentstore import *
from ainix_kernel.tests.testutils.torch_test_utils import eps_eq_at, torch_epsilon_eq
from pyrsistent import pmap

def test_latent_store():
    instance = TorchLatentStore(
        type_ind_to_latents=[
            LatentDataWrapper.construct(
                latents=torch.Tensor([[1, 2], [3, 1], [-3, 1], [2, -1]]),
                example_indxs=torch.LongTensor([0, 1, 2, 3]),
                y_inds=torch.LongTensor([2, 2, 2, 2]),
                impl_choices=torch.LongTensor([3, 1, 9, 1])
            )
        ],
        example_to_depths_to_ind_in_types=None
    )
    data, similarities = instance.get_n_nearest_latents(torch.Tensor([1, 1]), 0)
    print(similarities)
    assert torch.all(data.example_indxs == torch.LongTensor([1, 0, 3, 2]))
    assert torch.all(data.impl_choices == torch.LongTensor([1, 3, 1, 9]))
    torch_epsilon_eq(similarities, torch.Tensor([4, 3, 1, -2]))


@pytest.fixture()
def builder_store() -> TorchLatentStore:
    tc = TypeContext()
    ft = AInixType(tc, "FT")
    fo1 = AInixObject(tc, "FO1", "FT")
    fo2 = AInixObject(tc, "FO2", "FT", [AInixArgument(tc, "arg1", "BT", required=True)])
    bt = AInixType(tc, "BT")
    bo1 = AInixObject(tc, "BO1", "BT")
    AInixObject(tc, "BO2", "BT")
    tc.finalize_data()
    builder = TorchLatentStore.get_builder(tc.get_type_count(), 3)
    builder.add_example(
        0,
        ObjectChoiceNode(ft, ObjectNode(fo1, pmap()))
    )
    res: TorchLatentStore = builder.produce_result()
    assert res.example_to_depths_to_ind_to_types[0][0] == 0
    builder.add_example(
        1,
        ObjectChoiceNode(ft, ObjectNode(fo1, pmap()))
    )
    builder.add_example(
        2,
        ObjectChoiceNode(ft,
                         ObjectNode(fo2,
                                    pmap({"arg1": ObjectChoiceNode(bt, ObjectNode(bo1, pmap()))})
                                    )
                         )
    )
    store: TorchLatentStore = builder.produce_result()
    return store


def test_latent_store_builder(builder_store):
    assert len(builder_store.type_ind_to_latents) == 2
    assert torch_epsilon_eq(builder_store.type_ind_to_latents[1].example_indxs, [0, 1, 2])
    assert list(builder_store.example_to_depths_to_ind_to_types[1]) == [1]


def test_training(builder_store):
    trainer = builder_store.get_default_trainer()
    target = torch.Tensor([-0.4, 1, 2])
    type_ind = 1  # FT is type_ind 1
    example_id = 1
    dfs_depth = 0
    assert not torch_epsilon_eq(
        builder_store.get_latent_for_example(type_ind, example_id, dfs_depth),
        target, epsilon=1e-3
    )
    for i in range(100):
        print(builder_store.get_latent_for_example(type_ind, example_id, dfs_depth))
        trainer.update_value(1, 1, 0, target)
    assert torch_epsilon_eq(
        builder_store.get_latent_for_example(type_ind, example_id, dfs_depth),
        target, epsilon=1e-2
    )
