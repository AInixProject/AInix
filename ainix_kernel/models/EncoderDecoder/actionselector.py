""""""
import torch
from torch import nn


class ActionSelector(nn.Module):
    """An action selector chooses what to do for a ObjectChoice node. From a
    decoders it receives the a feature vector representing the current state.
    It then can decide what action to take (like PRODUCE-OBJECT, COPY, and
    eventually stuff like SUBPROC call and custom macros like insert-from-xargs
    insert-from-find-exec, or insert-from-stdpipe)"""
    def __init__(
        self,
        action_choose_projector: nn.Module
    ):
        pass

    def infer_predict(self, latent_vec: torch.Tensor) -> ActionResult:
        pass


    def train(self, latent_vec: torch.Tensor) -> ActionResult:
        pass


class ActionResult:
    pass


class CopyAction(ActionResult):
    pass


class ProduceObjectAction(ActionResult):
    pass


class SubprocAction(ActionResult):
    def __init__(self):
        raise NotImplemented()


class CustomAction(ActionResult):
    def __init__(self):
        raise NotImplemented()
