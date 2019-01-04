""""""
from abc import ABC
from typing import List
import attr

import torch
from torch import nn

from ainix_common.parsing.ast_components import ObjectNodeLike, CopyNode, ObjectNode, \
    AstObjectChoiceSet
from ainix_common.parsing.typecontext import AInixType, AInixObject
from ainix_kernel.models.multiforward import MultiforwardTorchModule, add_hooks


class ActionSelector(MultiforwardTorchModule, ABC):
    """An action selector chooses what to do for a ObjectChoice node. From a
    decoders it receives the a feature vector representing the current state.
    It then can decide what action to take (like PRODUCE-OBJECT, COPY, and
    eventually stuff like SUBPROC call and custom macros like insert-from-xargs
    insert-from-find-exec, or insert-from-stdpipe)"""
    @add_hooks
    def infer_predict(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        type_to_select: AInixType
    ) -> 'ActionResult':
        # for beam will could also have a max_return_count and extra_min_prob_return for 2nd val
        pass

    @add_hooks
    def forward_train(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        types_to_select: List[AInixType],
        expected: AstObjectChoiceSet,
        num_of_parents_with_copy_option: int,
    ) -> torch.Tensor:
        pass


class ActionResult:
    pass


@attr.s(frozen=True, auto_attribs=True)
class CopyAction(ActionResult):
    start: int
    end: int


@attr.s(frozen=True, auto_attribs=True)
class ProduceObjectAction(ActionResult):
    implementation: AInixObject


class SubprocAction(ActionResult):
    def __init__(self):
        raise NotImplemented()


class CustomAction(ActionResult):
    def __init__(self):
        raise NotImplemented()
