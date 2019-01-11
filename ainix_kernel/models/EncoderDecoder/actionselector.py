""""""
from abc import ABC, abstractmethod
from typing import List
import attr

import torch
from torch import nn

from ainix_common.parsing.ast_components import ObjectNodeLike, CopyNode, ObjectNode, \
    AstObjectChoiceSet, ObjectChoiceNode, AstIterPointer
from ainix_common.parsing.typecontext import AInixType, AInixObject
from ainix_kernel.models.multiforward import MultiforwardTorchModule, add_hooks


class ActionSelector(MultiforwardTorchModule, ABC):
    """An action selector chooses what to do for a ObjectChoice node. From a
    decoders it receives the a feature vector representing the current state.
    It then can decide what action to take (like PRODUCE-OBJECT, COPY, and
    eventually stuff like SUBPROC call and custom macros like insert-from-xargs
    insert-from-find-exec, or insert-from-stdpipe)"""
    @abstractmethod
    @add_hooks
    def infer_predict(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        type_to_select: AInixType
    ) -> 'ActionResult':
        # for beam will could also have a max_return_count and extra_min_prob_return for 2nd val
        pass

    @abstractmethod
    @add_hooks
    def forward_train(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        types_to_select: List[AInixType],
        expected: AstObjectChoiceSet,
        num_of_parents_with_copy_option: int,
        example_inds: List[int]
    ) -> torch.Tensor:
        pass

    def start_train_session(self):
        pass

    def end_train_session(self):
        pass

    def get_save_state_dict(self) -> dict:
        raise NotImplemented()


# Define classes for each kind of action you can take
# It is not exactly clear this strictly necessary as there is 1 to 1 mapping
# of these and subclasses of ObjectNodelike

class ActionResult:
    pass


@attr.s(frozen=True, auto_attribs=True)
class CopyAction(ActionResult):
    start: int
    end: int


@attr.s(frozen=True, auto_attribs=True)
class ProduceObjectAction(ActionResult):
    implementation: AInixObject

    def __attrs_post_init__(self):
        assert isinstance(self.implementation, AInixObject)


class SubprocAction(ActionResult):
    def __init__(self):
        raise NotImplemented()


class CustomAction(ActionResult):
    def __init__(self):
        raise NotImplemented()


def objectlike_to_action(node: ObjectNodeLike) -> ActionResult:
    if isinstance(node, ObjectNode):
        return ProduceObjectAction(node.implementation)
    elif isinstance(node, CopyNode):
        return CopyAction(node.start, node.end)
    else:
        raise ValueError()


class PathForceSpySelector(ActionSelector):
    """An action selector which logs the latent inputs and just always returns a
    fixed path. Used in decoder to get the latent states at each step"""
    def __init__(self, tree_path: ObjectChoiceNode):
        super().__init__()
        self.path_stack: List[AstIterPointer] = list(reversed(list(tree_path.depth_first_iter())))
        self.current_y_ind = 0
        self.lattents_log = []
        self.y_inds_log = []

    def infer_predict(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        type_to_select: AInixType
    ) -> ActionResult:
        cur = self.path_stack.pop().cur_node
        if isinstance(cur, ObjectChoiceNode):
            if cur.type_to_choose != type_to_select:
                raise RuntimeError("Types do not match on path. Some issue in calling")
            self.lattents_log.append(latent_vec)
            self.y_inds_log.append(self.current_y_ind)
            self.current_y_ind += 1
            # pop off the next objectnode since inference wont get called for that
            self.path_stack.pop()
            self.current_y_ind += 1
            # return along the force path
            return objectlike_to_action(cur.next_node)
        else:
            raise ValueError()

    @property
    def is_done(self) -> bool:
        return len(self.path_stack) == 0

    def forward_train(
        self,
        latent_vec: torch.Tensor,
        memory_tokens: torch.Tensor,
        types_to_select: List[AInixType],
        expected: AstObjectChoiceSet,
        num_of_parents_with_copy_option: int,
    ):
        raise ValueError("You can't train the spy selector")