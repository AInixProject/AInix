from ainix_kernel.model_util.operations import *
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq
import pytest


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


def test_pack_picks2():
    val = pack_picks(
        [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8])
        ],
        [torch.tensor([0, 2]), torch.tensor([]), torch.tensor([1, 2])]
    )
    assert torch_epsilon_eq(
        val,
        torch.tensor([1,3, 7, 8])
    )


def test_pack_picks3():
    val = pack_picks(
        [
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            torch.tensor([[3, 2, 1], [4, 3, 2]]),
            torch.tensor([[1, 8, 3], [4, 9, 6], [7, 0, 9]])
        ],
        [torch.tensor([0,2]), torch.tensor([]), torch.tensor([1,2])]
    )
    assert torch_epsilon_eq(
        val,
        torch.tensor([[1, 2, 3], [7, 8, 9], [4, 9, 6], [7, 0, 9]])
    )


def test_avg_pool():
    assert torch_epsilon_eq(
        avg_pool(
            torch.tensor([[[0., 3., 3.], [6., 3., 1.]]])
        ),
        torch.tensor([[3., 3., 2.]])
    )


def test_avg_pool2():
    assert torch_epsilon_eq(
        avg_pool(
            torch.tensor([
                [[0., 3., 3.], [6., 3., 1.]],
                [[8, 3., 4.], [4., 3., 6.]]
            ])
        ),
        torch.tensor([
            [3., 3., 2.],
            [6., 3., 5.]
        ])
    )


def test_avg_pool3():
    assert torch_epsilon_eq(
        avg_pool(
            torch.tensor([
                [[0., 3., 3.], [6., 3., 1.]],
                [[8, 3., 4.], [4., 3., 6.]]
            ]),
            torch.tensor([1, 2])
        ),
        torch.tensor([
            [0., 3., 3.],
            [6., 3., 5.]
        ])
    )


@pytest.mark.parametrize("use_cuda", (False, True))
def test_avg_pool4(use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA not available. Skipping")
    dvc = torch.device("cuda" if use_cuda else "cpu")
    assert torch_epsilon_eq(
        avg_pool(
            torch.tensor([
                [[0., 3., 3.], [6., 3., 1.], [0., 0., 0.]],
                [[8, 3., 4.], [4., 3., 6.], [6., 0., 11.]],
                [[8, 3., 4.], [4., 3., 6.], [0., 0., 0.]]
            ], device=dvc),
            torch.tensor([1, 3, 2], device=dvc)
        ),
        torch.tensor([
            [0., 3., 3.],
            [6., 2., 7.],
            [6., 3., 5.]
        ], device=dvc)
    )


def test_get_kernel_around0():
    assert torch_epsilon_eq(
        get_kernel_around(
            torch.tensor([[
                [1., 5, 7, 4, 6], [3., 6, 7, 2, 6]
            ]]),
            index=2,
            tokens_before_channels=False
        ),
        torch.tensor([[
            [5., 7, 4], [6, 7, 2]
        ]])
    )


def test_get_kernel_around():
    a, b, c = [3., 5], [1., 6], [1., 8]
    assert torch_epsilon_eq(
        get_kernel_around(
            torch.tensor([[
                [1., 2], a, b, c, [3, 5]
            ]]),
            index=2,
            tokens_before_channels=True
        ),
        torch.tensor([
            [a, b, c]
        ])
    )


def test_get_kernel_around_batched():
    a, b, c, d = [3., 5], [1., 6], [1., 8], [1., 7.]
    assert torch_epsilon_eq(
        get_kernel_around(
            torch.tensor([
                [[1., 2], a, b, c, [3, 5]],
                [a, a, d, c, c]
            ]),
            index=2,
            tokens_before_channels=True
        ),
        torch.tensor([
            [a, b, c],
            [a, d, c]
        ])
    )


def test_get_kernel_around_pad():
    a, b, c = [3., 5], [1., 6], [1., 8]
    assert torch_epsilon_eq(
        get_kernel_around(
            torch.tensor([[
                c, a, b, c, a
            ]]),
            index=0,
            tokens_before_channels=True
        ),
        torch.tensor([
            [[0, 0], c, a]
        ])
    )
