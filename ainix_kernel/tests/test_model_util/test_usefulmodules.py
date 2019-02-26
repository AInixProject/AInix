from ainix_kernel.model_util.usefulmodules import *
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq


def test_1dsame():
    mod = Conv1dSame(4, 4, 3, bias=False)
    v = mod(torch.rand(4, 4, 8))
    assert v.shape == (4, 4, 8)


def test_1dsame_check_weight():
    mod = Conv1dSame(1, 1, 3, bias=False)
    mod.set_weight(nn.Parameter(torch.Tensor([[[1., 1., 1.]]])))
    v = mod(torch.tensor([
        [[1., 2., 3., 4.]]
    ]))
    assert torch_epsilon_eq(
        v,
        torch.tensor([
            [[3., 6, 9, 7]]
        ])
    )
