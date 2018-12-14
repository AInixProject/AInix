from .torch_test_utils import *
import functools
import time

class SimpleRegression(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 8
        self.seq = torch.nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        if isinstance(x, list):
            # convert to batch tensor
            x = torch.stack(x)
        return self.seq(x)


def test_train_regression():
    torch.manual_seed(0)
    model = SimpleRegression()
    torch_train_tester(
        model,
        data=[((torch.Tensor([4]),), torch.Tensor([1])),
              ((torch.Tensor([-2]),), torch.Tensor([5]))
              ],
        comparer=functools.partial(torch_epsilon_eq, epsilon=1e-1),
        criterion=nn.L1Loss(),
        max_epochs=500,
        lr=1e-2
    )


def test_train_regression_batched():
    torch.manual_seed(0)
    model = SimpleRegression()
    torch_train_tester(
        model,
        data=[((torch.Tensor([0]),), torch.Tensor([0])),
              ((torch.Tensor([2]),), torch.Tensor([1])),
              ((torch.Tensor([6]),), torch.Tensor([2])),
              ((torch.Tensor([9]),), torch.Tensor([4])),
              ((torch.Tensor([-3]),), torch.Tensor([-3]))
              ],
        comparer=functools.partial(torch_epsilon_eq, epsilon=1e-1),
        criterion=nn.L1Loss(),
        max_epochs=5000,
        earyl_stop_patience=500,
        lr=1e-2,
        batch_size=2
    )


def test_early_stop():
    torch.manual_seed(0)
    model = SimpleRegression()
    start = time.time()
    torch_train_tester(
        model,
        data=[((torch.Tensor([[4]]),), torch.Tensor([[1]])),
              ((torch.Tensor([[-2]]),), torch.Tensor([[5]]))],
        comparer=functools.partial(torch_epsilon_eq, epsilon=1e-1),
        criterion=nn.L1Loss(),
        max_epochs=5000,
        lr=1e-2
    )
    end = time.time()
    assert (end - start) < 1



