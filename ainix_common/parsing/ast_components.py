import ainix_common.parsing.typecontext
from typing import Optional, Generator, \
    Tuple, MutableMapping, Mapping
from abc import ABC, abstractmethod
import attr
from pyrsistent import pmap


def indexable_repr_classify_type(type_name: str):
    return f"CLASSIFY_TYPE={type_name}"


def indexable_repr_object(object_name: str):
    return f"OBJECT={object_name}"


def convert_ast_to_indexable_repr():
    pass


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

    def depth_first_iter(self) -> Generator['AstNode', None, None]:
        """Iterates through tree starting at this node in a depth-first manner"""
        stack = [self]
        while stack:
            cur = stack.pop()
            yield cur
            stack.extend(reversed(list(cur.get_children())))

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
    def __init__(
        self,
        type_to_choose: ainix_common.parsing.typecontext.AInixType,
        frozen_choice: 'ObjectNode' = None
    ):
        self._type_to_choose = type_to_choose
        if type_to_choose is None:
            raise ValueError("Unable to create a ObjectChoiceNode with no type. If"
                             "the type is none, this node just show not exist.")
        self._verify_matching_types(frozen_choice)
        self._choice: 'ObjectNode' = frozen_choice
        self._is_frozen = frozen_choice is not None
        self._hash_cache = None

    def set_choice(self, new_choice: 'ObjectNode'):
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

    def freeze(self):
        self._choice.freeze()
        self._is_frozen = True

    def _verify_matching_types(self, new_choice: 'ObjectNode'):
        if new_choice is None:
            return
        if new_choice.implementation.type_name != self._type_to_choose.name:
            raise ValueError("Add unexpected choice as valid. Expected type " +
                             self.get_type_to_choose_name() + " got " +
                             new_choice.implementation.type_name)

    def get_type_to_choose_name(self) -> str:
        return self._type_to_choose.name

    @property
    def next_node(self) -> 'ObjectNode':
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


class ObjectNode(AstNode):
    def __init__(
        self,
        implementation: ainix_common.parsing.typecontext.AInixObject,
        frozen_arg_name_to_node: Mapping[str, ObjectChoiceNode] = None
    ):
        self._implementation = implementation
        if frozen_arg_name_to_node:
            self._arg_name_to_node = frozen_arg_name_to_node
            if len(self._arg_name_to_node) != len(implementation.children):
                raise ValueError("Unexepected number of children")
        else:
            self._arg_name_to_node = {}
        self._is_frozen = frozen_arg_name_to_node is not None
        self._hash_cache = None

    @property
    def implementation(self):
        return self._implementation

    def get_choice_node_for_arg(self, arg_name):
        return self._arg_name_to_node[arg_name]

    def freeze(self):
        if self._is_frozen:
            return
        self._is_frozen = True
        for node in self._arg_name_to_node.values():
            node.freeze()
        self._arg_name_to_node = pmap(self._arg_name_to_node)

    def set_arg_value(self, arg_name: str, new_node: ObjectChoiceNode):
        if self._is_frozen:
            raise ValueError("Can't set an arg of frozen node")
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
        val = self._implementation == other._implementation and \
               self._arg_name_to_node == other._arg_name_to_node
        return val


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


