from ainix_kernel.model_util.operations import *
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq


def test_manual_bincount():
    assert torch_epsilon_eq(
        manual_bincount(torch.Tensor([0, 1, 1, 0, 2])),
        [2, 2, 1]
    )


def test_manual_bincount_weighted():
    assert torch_epsilon_eq(
        manual_bincount(torch.Tensor([0, 1, 1, 0, 2]), torch.Tensor([1, 2, 1, 1, -2])),
        [2, 3, -2]
    )


def test_sparce_groupby_sum():
    reduced, group_keys = sparse_groupby_sum(
        torch.Tensor([1, 2, 1, 5]), torch.Tensor([0, 1, 1, 0]), sort_out_keys=True)
    assert torch_epsilon_eq(reduced, [6, 3])
    assert torch_epsilon_eq(group_keys, torch.Tensor([0, 1]))


def test_sparce_groupby_sum2():
    reduced, group_keys = sparse_groupby_sum(
        torch.Tensor([1, 2, 1, 5]), torch.Tensor([0, 1, 8, 0]), sort_out_keys=True)
    assert torch_epsilon_eq(reduced, [6, 2, 1])
    assert torch_epsilon_eq(group_keys, torch.Tensor([0, 1, 8]))


def test_pack_picks():
    val = pack_picks(
        [
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            torch.tensor([[3, 2, 1], [4, 3, 2]]),
            torch.tensor([[1, 8, 3], [4, 9, 6], [7, 0, 9]])
        ],
        [torch.tensor([0,2]), torch.tensor([1]), torch.tensor([1,2])]
    )
    assert torch_epsilon_eq(
        val,
        torch.tensor([[1, 2, 3], [7, 8, 9], [4, 3, 2], [4, 9, 6], [7, 0, 9]])
    )


def test_pack_picks1():
    val = pack_picks(
        [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8])
        ],
        [torch.tensor([0,2]), torch.tensor([1]), torch.tensor([1,2])]
    )
    assert torch_epsilon_eq(
        val,
        torch.tensor([1,3,5,7,8])
    )
