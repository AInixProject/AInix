from ainix_kernel.model_util.operations import *
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq


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
