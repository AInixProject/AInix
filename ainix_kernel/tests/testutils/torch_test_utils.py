from typing import Any, Sequence, Tuple
from torch import nn
import torch

def torch_epsilon_eq(a, b, epsilon=1e-12):
    """Test if two float tensors are equal within some epsilon"""
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), epsilon))

def torch_train_tester(
    model: nn.Module,
    data: Sequence[Tuple[Any, Any]],
    comparer = lambda x, y: x == y,
    y_extractor_train=lambda y: y,
    y_extractor_eval=lambda y: y,
    criterion=lambda x, y: y,
    num_epochs=100,
    lr=1e-3
):
    """A generic util for training a nn.Model to get some expected output.

    Args:
        model: The model to train and eval
        data: A sequence of ((x, ...), y) tuples. The first value of the tuple is arg
            expanded and passed into the forward of the model. The second value
            is the expected outputs to the model.
        comparer: A callable which returns whether the x matches y
        y_extractor_train: A callable which takes the output of the forward of
            the model and converts into a usable form for the criterion. This
            can be used in case the forward of the model returns something
            like a tuple and only one of which relevant to the loss. Defaults
            to an identity.
        y_extractor_eval: A callable that takes the output of the forward of the
            model and converts it a form to pass into the comparer. Defaults to
            an identity func.
        criterion: A criterion to apply on the extracted y output of the model.
            It defaults to a function which only returns the extracted y, which
            can be used if the model already returns a backprop-able loss.
        num_epochs: num epochs to train for
        lr: the learning rate to use in adam optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Train
    model.train()
    for epoch in range(num_epochs):
        for x, y in data:
            optimizer.zero_grad()
            y_hat = y_extractor_train(model(*x))
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
    # eval
    model.eval()
    for x, y in data:
        y_hat = y_extractor_eval(model(*x))
        assert comparer(y_hat, y), f"Expected {y}. Predicted {y_hat}"

