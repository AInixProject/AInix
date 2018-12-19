import collections

import ainix_common.parsing.typecontext
from typing import Optional, Generator, \
    Tuple, MutableMapping, Mapping, List
from abc import ABC, abstractmethod
import attr
from pyrsistent import pmap


def indexable_repr_classify_type(type_name: str):
    return f"CLASSIFY_TYPE={type_name}"


def indexable_repr_object(object_name: str):
    return f"OBJECT={object_name}"


def convert_ast_to_indexable_repr():
    pass


@attr.s(auto_attribs=True, frozen=True)
class AstIterPointer:
    """A pointer into an AST. This can be used while iterating through an AST.
    It keeps track of where it is in the tree. AST nodes don't store a reference
    to their parent, so this class helps get back to the root of a node. This can
    also be used to make a copy of tree with substructure sharing for common parts"""
    cur_node: 'AstNode'
    parent: Optional['AstIterPointer']
    parent_child_ind: Optional[int]

    def dfs_get_next(self) -> Optional['AstIterPointer']:
        """Gets the next element as viewed as depth first search from root"""
        next_src = self
        parent_i = self.parent_child_ind
        on_indx = 0
        while next_src:
            n = next_src.cur_node.get_nth_child(on_indx, True)
            if n:
                return AstIterPointer(n, next_src, on_indx)
            else:
                if next_src.parent_child_ind is not None:
                    on_indx = next_src.parent_child_ind + 1
                else:
                    return None
                next_src = next_src.parent


class AstNode(ABC):
    #def __init__(self, parent: Optional['AstNode']):
    #    self.parent = parent
    #def __init__(self):
    #    self._is_frozen = False

    def dump_str(self, indent=0):
        return "  " * indent + str(self) + "\n"

    @abstractmethod
    def get_children(self) -> Generator['AstNode', None, None]:
        """Returns all descendants of this node"""
        pass

    @abstractmethod
    def get_nth_child(self, n, none_if_out_of_bounds = False) -> Optional['AstNode']:
        """Get the nth child. Note this might be different than the nth element
        in the result of get_children() as that only returns set children.
        Might raise a IndexError unless none_if_out_of_bounds is true.
        """
        pass

    def path_clone(
        self,
        unfreeze_path: List['AstNode'] = None
    ) -> Tuple['AstNode', Optional[List['AstNode']]]:
        """Creates a deep copy of this node. In general frozen nodes are simply
        returned as a reference and mutable nodes are instantiated to a new object.
        Nodes along the path parameter are returned as mutable copies regardless
        of whether they are frozen or not. This can be used to do substructure
        sharing when you want to mutate along some path.
        Args:
            unfreeze_path: the path down which we will always create an
                unfrozen copy. From the user's perspective this method will
                probably called at the root of a tree. In which case self should
                be unfreeze_path[0]. The path is always in root-to-leaf order
                taking only branch at a time. If the unfreeze path is non-none,
                but the last the last element in the list is not a leaf node,
                then cloning halts at the element and it becomes a leaf.

        Returns:
            cloned_version: The cloned version of the node.
            new_path: A list representing the same path as the unfreeze_path passed in except
                that it is now the equivolent versions in the new AST
        """
        raise NotImplemented()

    def depth_first_iter(self) -> Generator['AstIterPointer', None, None]:
        """Iterates through tree starting at this node in a depth-first manner"""
        cur = AstIterPointer(self, None, None)
        while cur:
            yield cur
            cur = cur.dfs_get_next()

    @abstractmethod
    def freeze(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    def __ne__(self, other):
        return not self.__eq__(other)


class ObjectChoiceNode(AstNode):
    """Given a certain type, represents a selection amoung the available implementations."""
    def __init__(
        self,
        type_to_choose: ainix_common.parsing.typecontext.AInixType,
        frozen_choice: 'ObjectNodeLike' = None
    ):
        self._type_to_choose = type_to_choose
        if type_to_choose is None:
            raise ValueError("Unable to create a ObjectChoiceNode with no type. If"
                             "the type is none, this node just show not exist.")
        self._verify_matching_types(frozen_choice)
        self._choice: 'ObjectNodeLike' = frozen_choice
        self._is_frozen = frozen_choice is not None
        self._hash_cache = None

    def set_choice(self, new_choice: 'ObjectNodeLike'):
        if self._is_frozen:
            raise ValueError("Cannot mutate frozen node")
        self._verify_matching_types(new_choice)
        self._choice = new_choice

    @property
    def type_to_choose(self):
        return self._type_to_choose

    @property
    def choice(self):
        return self._choice

    @property
    def is_frozen(self):
        return self._is_frozen

    @property
    def copy_was_chosen(self) -> bool:
        if self._choice is None:
            raise ValueError("Not chosen yet")
        return isinstance(self._choice, CopyNode)

    def freeze(self):
        self._choice.freeze()
        self._is_frozen = True

    def _verify_matching_types(self, new_choice: 'ObjectNodeLike'):
        if new_choice is None:
            return
        if not new_choice.is_of_type(self._type_to_choose):
            raise ValueError("Add unexpected choice as valid.")

    def get_type_to_choose_name(self) -> str:
        return self._type_to_choose.name

    @property
    def next_node(self) -> 'ObjectNodeLike':
        return self._choice

    @property
    def type_context(self) -> ainix_common.parsing.typecontext.TypeContext:
        return self._type_to_choose.type_context

    def get_chosen_impl_name(self) -> str:
        return self._choice._implementation.name

    def __str__(self):
        s = "<ObjectChoiceNode for " + str(self.get_type_to_choose_name())
        s += " valid_choices=" + str(self._choice)
        s += ">"
        return s

    def dump_str(self, indent=0):
        indent_str = "  " * indent
        s = indent_str + "<ObjectChoiceNode type " + self.get_type_to_choose_name() + "> {\n"
        s += self._choice.dump_str(indent + 2)
        s += indent_str + "}\n"
        return s

    def indexable_repr(self) -> str:
        repr = indexable_repr_classify_type(self.get_type_to_choose_name())
        repr += f" O[O {self._choice.indexable_repr()} O]O"
        return repr

    def get_children(self) -> Generator[AstNode, None, None]:
        if self._choice is not None:
            yield self._choice

    def get_nth_child(self, n, none_if_out_of_bounds = False) -> Optional['AstNode']:
        if n == 0:
            return self._choice
        else:
            if none_if_out_of_bounds:
                return None
            raise IndexError()

    def path_clone(
        self,
        unfreeze_path: List['AstNode'] = None
    ) -> Tuple['AstNode', Optional[List['AstNode']]]:
        """See docstring on AstNode.path_clone()"""
        on_unfreeze_path = unfreeze_path is not None and id(self) == id(unfreeze_path[0])
        if self._is_frozen and not on_unfreeze_path:
            return self, None
        clone = ObjectChoiceNode(self.type_to_choose)
        next_unfreeze_path = unfreeze_path[1:] if on_unfreeze_path else None
        # If we have reached the end of the path the path list we "stop_early" and
        # force the ourself to become a leaf on the new tree.
        stop_early_on_path = next_unfreeze_path is not None and len(next_unfreeze_path) == 0
        if stop_early_on_path:
            return clone, [clone]
        #
        new_path = None
        if self._choice is not None:
            clone._choice, new_path = self._choice.path_clone(next_unfreeze_path)
        if on_unfreeze_path:
            if not new_path:
                new_path = []
            new_path.insert(0, clone)
        return clone, new_path

    def __repr__(self):
        return f"<ObjecChoiceNode{'(F)' if self._is_frozen else ''} {self.type_to_choose.name}>"

    def __hash__(self):
        if self._hash_cache:
            return self._hash_cache
        if not self._is_frozen:
            raise ValueError("Must be frozen to hash")
        self._hash_cache =  hash((self._type_to_choose, self._choice))
        return self._hash_cache

    def __eq__(self, other):
        if other is None:
            return False
        if id(self) == id(other):
            return True
        if not hasattr(other, "_type_to_choose"):
            return False
        return self._type_to_choose == other._type_to_choose and \
               self._choice == other._choice


class ChildlessObjectNode:
    def __init__(
        self,
        implementation: ainix_common.parsing.typecontext.AInixObject,
        chosen_type_names: Tuple[str, ...]
    ):
        self._implementation = implementation
        self._chosen_type_names = chosen_type_names

    @property
    def implementation(self):
        return self.implementation

    @property
    def chosen_type_names(self):
        return self._chosen_type_names

    def __hash__(self):
        return hash((self._implementation, self._chosen_type_names))

    def __eq__(self, other):
        return self._implementation == other._implementation and \
                self._chosen_type_names == other._chosen_type_names

    def __ne__(self, other):
        return not self.__eq__(other)


class ObjectNodeLike(AstNode, ABC):
    """Something that is pickable in an ObjectChoice"""
    @abstractmethod
    def is_of_type(self, type_: ainix_common.parsing.typecontext.AInixType) -> bool:
        """Checks whether this is object-like node is a valid choice for a certain type"""
        pass


class ObjectNode(ObjectNodeLike):
    def __init__(
        self,
        implementation: ainix_common.parsing.typecontext.AInixObject,
        frozen_arg_name_to_node: Mapping[str, ObjectChoiceNode] = None
    ):
        self._implementation = implementation
        if frozen_arg_name_to_node is not None:
            self._arg_name_to_node = frozen_arg_name_to_node
            if len(self._arg_name_to_node) != len(implementation.children):
                raise ValueError("Unexepected number of children")
            assert isinstance(self._arg_name_to_node, collections.Hashable)
            assert not isinstance(self._arg_name_to_node, dict)
        else:
            self._arg_name_to_node = {arg.name: None for arg in implementation.children}
        self._is_frozen = frozen_arg_name_to_node is not None
        self._hash_cache = None

    @property
    def implementation(self) -> ainix_common.parsing.typecontext.AInixObject:
        return self._implementation

    def __repr__(self):
        return f"<ObjectNode{'(F)' if self._is_frozen else ''} {self._implementation.name}>"

    def get_choice_node_for_arg(self, arg_name) -> 'ObjectChoiceNode':
        return self._arg_name_to_node[arg_name]

    def freeze(self):
        if self._is_frozen:
            return
        self._is_frozen = True
        for node in self._arg_name_to_node.values():
            node.freeze()
        self._arg_name_to_node = pmap(self._arg_name_to_node)

    @property
    def is_frozen(self):
        return self._is_frozen

    def set_arg_value(self, arg_name: str, new_node: ObjectChoiceNode):
        if self._is_frozen:
            raise ValueError("Can't set an arg of frozen node")
        if arg_name not in self._arg_name_to_node:
            raise ValueError(f"Arg name {arg_name} not recognized for {self.implementation}."
                             f"Options are {list(self._arg_name_to_node.keys())}")
        self._arg_name_to_node[arg_name] = new_node

    def __str__(self):
        return "<ObjectNode for " + self._implementation.name + ">"

    def dump_str(self, indent=0):
        indent_str = "  " * indent
        s = indent_str + "<ObjectNode obj " + self._implementation.name + "> {\n"
        s += indent_str + "  next_type_choices: {\n"
        for arg in self.implementation.children:
            if arg.name in self._arg_name_to_node:
                s += self._arg_name_to_node[arg.name].dump_str(indent + 2)
            else:
                s += indent_str + "    missing " + arg.name + "\n"
        s += indent_str + "  }\n"
        s += indent_str + "}\n"
        return s

    def indexable_repr(self) -> str:
        repr = indexable_repr_object(self._implementation.name)
        repr += " ARGS"
        for arg in self._implementation.children:
            repr += ' ARG=' + self._implementation.name + "::" + arg.name
            arg_node = self._arg_name_to_node[arg.name]
            repr += ' T[T'
            repr += ' ' + arg_node.indexable_repr()
            repr += ' T]T'
        repr += " ENDARGS"
        return repr

    def get_children(self) -> Generator[AstNode, None, None]:
        yield from (self._arg_name_to_node[arg.name]
                    for arg in self._implementation.children
                    if arg.name in self._arg_name_to_node)

    def get_nth_child(self, n, none_if_out_of_bounds = False) -> Optional['AstNode']:
        try:
            return self._arg_name_to_node[self._implementation.children[n].name]
        except IndexError:
            if none_if_out_of_bounds:
                return None
            else:
                raise

    def as_childless_node(self) -> ChildlessObjectNode:
        choices = tuple([self._arg_name_to_node[arg.name].get_chosen_impl_name()
                         for arg in self._implementation.children])
        return ChildlessObjectNode(self._implementation, choices)

    def __hash__(self):
        if self._hash_cache:
            return self._hash_cache
        if not self._is_frozen:
            raise ValueError("Can't hash non-frozen node")
        self._hash_cache = hash((self.implementation, self._arg_name_to_node))
        return self._hash_cache

    def __eq__(self, other):
        if not hasattr(other, "_implementation"):
            return False
        val = self._implementation == other._implementation and \
               self._arg_name_to_node == other._arg_name_to_node
        return val

    def is_of_type(self, type_: ainix_common.parsing.typecontext.AInixType) -> bool:
        return self._implementation.type_name == type_.name

    def path_clone(
        self,
        unfreeze_path: List['AstNode'] = None
    ) -> Tuple['ObjectNode', Optional[List['AstNode']]]:
        """See docstring on AstNode.path_clone()"""
        on_unfreeze_path = unfreeze_path is not None and id(self) == id(unfreeze_path[0])
        if self._is_frozen and not on_unfreeze_path:
            return self, None
        clone = ObjectNode(self.implementation)
        next_unfreeze_path = unfreeze_path[1:] if on_unfreeze_path else None
        # If we have reached the end of the path the path list we "stop_early" and
        # force the ourself to become a leaf on the new tree.
        stop_early_on_path = next_unfreeze_path is not None and len(next_unfreeze_path) == 0
        if stop_early_on_path:
            return clone, [clone]
        #
        return_back_path = None
        for arg in self.implementation.children:
            if arg.name in self._arg_name_to_node:
                arg_clone, arg_copy_path = self._arg_name_to_node[arg.name].path_clone(
                    next_unfreeze_path
                )
                clone.set_arg_value(arg.name, arg_clone)
                if arg_copy_path:
                    return_back_path = arg_copy_path
        if on_unfreeze_path:
            if not return_back_path:
                return_back_path = []
            return_back_path.insert(0, clone)
        return clone, return_back_path


class CopyNode(ObjectNodeLike):
    def __init__(
        self,
        copy_type: 'ainix_common.parsing.typecontext.AInixType',
        start: int = None,
        end: int = None
    ):
        self._copy_type = copy_type
        self._start = start
        self._end = end
        self._is_frozen = start is not None and end is not None
        self._hash_cache = None

    def freeze(self):
        if self._is_frozen:
            return
        if self._start is None:
            raise ValueError("Cannot freeze copy node with unset start")
        if self._end is None:
            raise ValueError("Cannot freeze copy node with unset end")
        self._is_frozen = True

    def get_children(self) -> Generator['AstNode', None, None]:
        yield from iter([])

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self._copy_type == other._copy_type and \
            self._start == other._start and \
            self._end == other._end

    def __hash__(self):
        if self._hash_cache:
            return self._hash_cache
        if not self._is_frozen:
            raise ValueError("Cannot hash CopyNode that is not frozen")
        self._hash_cache = hash((self._copy_type.name, self._start, self._end))
        return self._hash_cache

    def is_of_type(self, type_: ainix_common.parsing.typecontext.AInixType) -> bool:
        return type_ == self._copy_type


class AstSet:
    def __init__(self, parent: Optional['AstSet']):
        self._parent = parent
        self._is_frozen = False

    @abstractmethod
    def freeze(self):
        pass

    def get_depth(self):
        cur = self
        depth = 0
        while cur._parent:
            depth += 1
            cur = cur._parent
        return depth

    def get_root(self) -> 'AstNode':
        return self if self._parent is None else self.get_root()

    @abstractmethod
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self.__eq__(other)

    @abstractmethod
    def __hash__(self):
        if not self._is_frozen:
            raise ValueError("Unable to hash non-frozen AstSet")

    @abstractmethod
    def is_node_known_valid(self, node: AstNode) -> bool:
        pass

    @abstractmethod
    def add(self, node: AstNode, known_as_valid: bool,
            weight: float, probability_valid: float) -> None:
        pass


class ArgsSetData:
    def __init__(
        self,
        implementation: ainix_common.parsing.typecontext.AInixObject,
        part_of: 'ObjectNodeSet'
    ):
        self.arg_to_choice_set: Mapping[str, 'AstObjectChoiceSet'] = {
            arg.name: AstObjectChoiceSet(arg.next_choice_type, part_of)
                      if arg.next_choice_type else None
            for arg in implementation.children
        }
        self._max_probability: float = 0
        self._is_known_valid: bool = False
        self._max_weight: float = 0
        self._is_frozen: bool = False

    def freeze(self):
        self._is_frozen = True
        for node in self.arg_to_choice_set.values():
            node.freeze()
        self.arg_to_choice_set = pmap(self.arg_to_choice_set)

    def __hash__(self):
        if not self._is_frozen:
            raise ValueError("Object must be frozen to be hashed")
        return hash((self.arg_to_choice_set, self._max_probability,
                     self._is_known_valid, self._max_weight))

    def __eq__(self, other):
        return self.arg_to_choice_set == other.arg_to_choice_set and \
               self._max_probability == other._max_proobability and \
               self._is_known_valid == other._is_known_valid and \
               self._max_weight == other._max_weight and \
               self._is_frozen == other._is_frozen

    def __ne__(self, other):
        return not self.__eq__(other)

    def add_from_other_data(self, node: ObjectNode, new_probability,
                            new_is_known_valid_, new_weight):
        if self._is_frozen:
            raise ValueError("Can't mutate frozen data")
        for arg in node._implementation.children:
            self.arg_to_choice_set[arg.name].add(node._arg_name_to_node[arg.name],
                                                 new_is_known_valid_, new_weight, new_probability)
        self._max_probability = max(self._max_probability, new_probability)
        self._is_known_valid = self._is_known_valid or new_is_known_valid_
        self._max_weight = max(self._max_weight, new_weight)


class ObjectNodeSet(AstSet):
    def __init__(
        self,
        implementation: ainix_common.parsing.typecontext.AInixObject,
        parent: Optional[AstSet]
    ):
        super().__init__(parent)
        self._implementation = implementation
        self.data: MutableMapping[ChildlessObjectNode, ArgsSetData] = {}

    def freeze(self):
        if self._is_frozen:
            return
        self._is_frozen = True
        for d in self.data.values():
            d.freeze()
        self.data = pmap(self.data)

    def get_arg_set_data(self, arg_selection: ChildlessObjectNode) -> ArgsSetData:
        return self.data.get(arg_selection, None)

    def _verify_impl_of_new_node(self, node):
        if node.implementation != self._implementation:
            raise ValueError(
                f"Cannot add node with implementation {node.implementation} into"
                f" set of {self._implementation} objects")

    def add(self, node: ObjectNode, known_as_valid: bool, weight: float, probability_valid: float):
        self._verify_impl_of_new_node(node)
        childless_args = node.as_childless_node()
        if childless_args not in self.data:
            self.data[childless_args] = ArgsSetData(node._implementation, self)
        self.data[childless_args].add_from_other_data(
            node, probability_valid, known_as_valid, weight)

    def is_node_known_valid(self, node: ObjectNode) -> bool:
        childless_args = node.as_childless_node()
        if childless_args not in self.data:
            return False
        node_data = self.data[childless_args]
        if not node_data._is_known_valid:
            return False
        for arg in node.implementation.children:
            if arg.name not in node_data.arg_to_choice_set:
                return False
            if not node_data.arg_to_choice_set[arg.name].is_node_known_valid(
                    node.get_choice_node_for_arg(arg.name)):
                return False
        return True

    def __hash__(self):
        super().__hash__()
        return hash((self._implementation, self.data))

    def __eq__(self, other):
        return self._implementation == other._implementation and \
               self.data == other.data


@attr.s(auto_attribs=True, frozen=True)
class ImplementationData:
    next_node: ObjectNodeSet
    max_probability_valid: float
    known_as_valid: bool
    max_weight: float


class AstObjectChoiceSet(AstSet):
    def __init__(
        self,
        type_to_choose: ainix_common.parsing.typecontext.AInixType,
        parent: Optional[AstSet]
    ):
        super().__init__(parent)
        self._type_to_choose = type_to_choose
        self._impl_name_to_data: MutableMapping[str, 'ImplementationData'] = {}
        self._max_weight = 0
        self._hash_cache = None

    @property
    def type_to_choose(self):
        return self._type_to_choose

    @property
    def type_to_choose_name(self):
        return self._type_to_choose.name

    def get_next_node_for_choice(self, impl_name_chosen: str) -> Optional[ObjectNodeSet]:
        if impl_name_chosen not in self._impl_name_to_data:
            return None
        return self._impl_name_to_data[impl_name_chosen].next_node

    def freeze(self):
        self._is_frozen = True
        for n in self._impl_name_to_data.values():
            if n.next_node:
                n.next_node.freeze()
        self._impl_name_to_data = pmap(self._impl_name_to_data)

    def add(
        self,
        node: ObjectChoiceNode,
        known_as_valid: bool,
        weight: float,
        probability_valid: float
    ):
        if self._is_frozen:
            raise ValueError("Cannot add to frozen AstObjectChoiceSet")
        if node.type_to_choose.name != self._type_to_choose.name:
            raise ValueError(f"Attempting to add node of ObjectChoiceNode of type "
                             f"{node.type_to_choose.name} into AstObjectChoice set "
                             f"of type {self.type_to_choose.name}")

        existing_data = self._impl_name_to_data.get(node.get_chosen_impl_name(), None)
        if existing_data:
            new_weight = max(weight, existing_data.max_weight)
            if existing_data.max_probability_valid is None:
                new_probability_valid = None
            else:
                new_probability_valid = max(
                    probability_valid, existing_data.max_probability_valid
                )
            new_known_valid = known_as_valid or existing_data.known_as_valid
        else:
            new_weight = weight
            new_probability_valid = probability_valid
            new_known_valid = known_as_valid

        # figure out the data's next node
        if existing_data and existing_data.next_node is not None:
            next_node = existing_data.next_node
        elif node.next_node:
            next_node = ObjectNodeSet(node.next_node.implementation, self)
        else:
            next_node = None
        new_data = ImplementationData(next_node, new_probability_valid, new_known_valid, new_weight)
        self._impl_name_to_data[node.get_chosen_impl_name()] = new_data
        if next_node:
            next_node.add(node.next_node, known_as_valid, weight, probability_valid)

    def is_node_known_valid(self, node: ObjectChoiceNode) -> bool:
        if node is None:
            return False
        if node.get_chosen_impl_name() not in self._impl_name_to_data:
            return False
        data = self._impl_name_to_data[node.get_chosen_impl_name()]
        if not data.known_as_valid:
            return False
        if data.next_node is None:
            return node.next_node is None
        return data.next_node.is_node_known_valid(node.next_node)

    def is_known_choice(self, choose_name: str):
        if choose_name not in self._impl_name_to_data:
            return False
        return self._impl_name_to_data[choose_name].known_as_valid

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self._type_to_choose == other._type_to_choice and \
               self._impl_name_to_data == other._impl_name_to_data and \
               self._max_weight == other._max_weight

    def __hash__(self):
        if self._hash_cache:
            return self._hash_cache
        super().__hash__()
        hash_val = hash((self._type_to_choose, self._impl_name_to_data))
        self._hash_cache = hash_val
        return hash_val


