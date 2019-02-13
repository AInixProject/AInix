"""This code tries to address the desire to have different forwards with
different signatures and supporting static analysis / IDE hinting.

Example:
    class MyModule(MultiforwardTorchModule):
        @add_hooks
        def forward_train(
            hidden_state: torch.Tensor,
            teacher_force_seq: List[str]
            # training specific args...
        ) -> torch.Tensor:
            # ....
            return loss

        @add_hooks
        def forward_inference(
            hidden_state: torch.Tensor,
            beam_size: int
            # inference specific args...
        ) -> List[str]:
            # ....
            return result

    # mod = MyModule()
    # instead of this: mod(foo, bar)
    # we can do this: mod.forward_train(foo, bar)
    # and still have the forward/backwards hooks called
"""
from typing import Callable
import torch.nn


def add_hooks(new_forward_func):
    """A decorator for for methods inside a MultifowardTorchModule to make a
    forward act like a forward call (still calling the forwards/backwards
    hooks)"""
    def wrapper(self: MultiforwardTorchModule, *args, **kwargs):
        return self(new_forward_func, self, *args, **kwargs)
    return wrapper


class MultiforwardTorchModule(torch.nn.Module):
    """Wraps nn.Module to work with add_forward hooks. Instead of overriding
    forward and calling this module with __call__, you can just use the
    add_hooks on methods that act like a forward"""
    def forward(self, actual_forward: Callable, *args, **kwargs):
        """Calls the value passed in from the annotation. This should not be
        overridden (unless you want to create something that happens on all
        your forwards somewhat like a forward hook.)"""
        return actual_forward(*args, **kwargs)
