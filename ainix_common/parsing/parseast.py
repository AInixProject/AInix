from collections import defaultdict
import parse_primitives
import typecontext
from typing import List, Dict, Optional, Type, Union, Generator, \
    Tuple, MutableMapping, Mapping
from attr import attrs, attrib
from abc import ABC, abstractmethod, abstractproperty
from functools import lru_cache
import attr
from pyrsistent import pmap, PRecord, field


def indexable_repr_classify_type(type_name: str):
    return f"CLASSIFY_TYPE={type_name}"


def indexable_repr_object(object_name: str):
    return f"OBJECT={object_name}"


def convert_ast_to_indexable_repr():
    pass


class AstNode(ABC):
    #def __init__(self, parent: Optional['AstNode']):
    #    self.parent = parent

    def dump_str(self, indent=0):
        return "  " * indent + str(self) + "\n"

    #def get_depth(self):
    #    cur = self
    #    depth = 0
    #    while cur.parent:
    #        depth += 1
    #        cur = cur.parent
    #    return depth

    #def get_root(self) -> 'AstNode':
    #    return self if self.parent is None else self.get_root()

    @abstractmethod
    def get_children(self) -> Generator['AstNode', None, None]:
        """Returns all descendants of this node"""
        pass

    def depth_first_iter(self) -> Generator['AstNode', None, None]:
        """Iterates through tree starting at this node in a depth-first manner"""
        yield from map(self.depth_first_iter, self.get_children())

    #@abstractmethod
    #def __eq__(self, other):
    #    pass

    #@abstractmethod
    #def __hash__(self):
    #    pass

    #def __ne__(self, other):
    #    return not self.__eq__(self, other)


class ObjectChoiceLikeNode(AstNode):
    @abstractmethod
    def get_type_to_choose_name(self) -> str:
        pass

    @abstractproperty
    def type_context(self) -> typecontext.TypeContext:
        pass

    @abstractmethod
    def get_chosen_impl_name(self) -> str:
        pass

    @abstractproperty
    def chosen_object_node(self) -> 'ObjectNode':
        pass

    @abstractmethod
    def indexable_repr(self) -> str:
        pass

@attr.s(auto_attribs=True, frozen=True, cache_hash=True)
class ObjectChoiceNode(ObjectChoiceLikeNode):
    type_to_choose: typecontext.AInixType
    _choice: 'ObjectNode'

    def __attrs_post_init__(self):
        if self._choice.implementation.type_name != self.type_to_choose.name:
            raise ValueError("Add unexpected choice as valid. Expected type " +
                             self.get_type_to_choose_name() + " got " +
                             self._choice.implementation.type_name)

    def get_type_to_choose_name(self) -> str:
        return  self.type_to_choose.name

    @property
    def chosen_object_node(self):
        return self._choice

    @property
    def type_context(self) -> typecontext.TypeContext:
        return self.type_to_choose.type_context

    def get_chosen_impl_name(self) -> str:
        return self._choice.implementation.name

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
        repr += f" O[O {self.choice.indexable_repr()} O]O"
        return repr

    def get_children(self) -> Generator[AstNode, None, None]:
        assert self._choice is not None
        yield self._choice


@attr.s(auto_attribs=True, frozen=True, cache_hash=True)
class ArgPresentChoiceNode(ObjectChoiceLikeNode):
    argument: typecontext.AInixArgument
    is_present: bool
    obj_choice_node: Optional['ObjectChoiceNode']

    def __attrs_post_init__(self):
        if self.is_present and self.obj_choice_node is None and \
                self.argument.type is not None:
            raise ValueError(f"Argument {self.argument.name} requires an value, yet"
                             f" a ArgPresentChoice node constructed with no"
                             f"obj_choice_node")
        if self.obj_choice_node is not None and self.argument.type is None:
            raise ValueError(f"Argument {self.argument.name} has no type, yet"
                             f"a ArgPresentChoice node constructed with an"
                             f"ObjectChoiceNode.")

    def get_type_to_choose_name(self) -> str:
        return self.argument.present_choice_type_name

    @property
    def chosen_object_node(self):
        if not self.is_present:
            return None
        return self.obj_choice_node

    @property
    def type_context(self) -> typecontext.TypeContext:
        return self.argument._type_context

    def get_chosen_impl_name(self) -> str:
        return self.argument.is_present_name if self.is_present else \
            self.argument.not_present_name

    #def set_valid_choice_by_name(self, obj_name: str) -> 'AstNode':
    #    if obj_name == self.argument.is_present_name:
    #        return self.set_choice(True)
    #    elif obj_name == self.argument.not_present_name:
    #        return self.set_choice(False)
    #    else:
    #        raise ValueError(f"tried to set ArgPresentChoiceNode with invalid"
    #                         f" object name {obj_name}")

    def __str__(self):
        return "<ArgPresentChoiceNode for " + str(self.argument) + \
            ". " + str(self.is_present) + ">"

    def dump_str(self, indent=0):
        indent_str = "  " * indent
        s = indent_str + "<ArgPresentChoiceNode for " + self.argument.name +\
            ". " + str(self.is_present) + "> {\n"
        if self.obj_choice_node:
            s += self.obj_choice_node.dump_str(indent + 1)
        s += indent_str + "}\n"
        return s

    def indexable_repr(self) -> str:
        # Just pretend to be a type in index output
        # TODO (DNGros): This is very not clean. Optional args should probably define own types
        repr = indexable_repr_classify_type(self.get_type_to_choose_name())
        repr += f" O[O {indexable_repr_object(self.get_chosen_impl_name())} ARGS"
        if self.is_present and self.obj_choice_node:
            repr += f" ARG={self.argument.is_present_name}::VAL"
            repr += f" T[T {self.obj_choice_node.indexable_repr()} T]T"
        repr += " ENDARGS O]O"
        return repr

    def get_children(self) -> Generator[AstNode, None, None]:
        if self.obj_choice_node:
            yield self.obj_choice_node


@attr.s(auto_attribs=True, frozen=True)
class ChildlessObjectNode():
    _implementation: typecontext.AInixObject
    _chosen_type_names: Tuple[str, ...]


@attr.s(auto_attribs=True, frozen=True, cache_hash=True)
class ObjectNode(AstNode):
    implementation: typecontext.AInixObject
    arg_name_to_node: Mapping[str, ObjectChoiceLikeNode]

    #def _get_default_arg_name_to_node_dict(self):
    #    out = {}
    #    for arg in self.implementation.children:
    #        if not arg.required:
    #            out[arg.name] = ArgPresentChoiceNode(arg, self)
    #        else:
    #            out[arg.name] = ObjectChoiceNode(arg.type, self)
    #    return out

    #def set_arg_present(self, arg: typecontext.AInixArgument) -> Optional[ObjectChoiceLikeNode]:
    #    if not arg.required:
    #        already_made_present_choice = self.arg_name_to_node[arg.name]
    #        return already_made_present_choice.set_choice(True)
    #    elif arg.type is not None:
    #        return self.arg_name_to_node[arg.name]
    #    else:
    #        return None

    def __str__(self):
        return "<ObjectNode for " + self.implementation.name + ">"

    def dump_str(self, indent=0):
        indent_str = "  " * indent
        s = indent_str + "<ObjectNode obj " + self.implementation.name + "> {\n"
        s += indent_str + "  next_type_choices: {\n"
        for name, choice in self.arg_name_to_node.items():
            s += choice.dump_str(indent + 2)
        s += indent_str + "  }\n"
        s += indent_str + "}\n"
        return s

    def indexable_repr(self) -> str:
        repr = indexable_repr_object(self.implementation.name)
        repr += " ARGS"
        for arg in self.implementation.children:
            repr += ' ARG=' + self.implementation.name + "::" + arg.name
            arg_node = self.arg_name_to_node[arg.name]
            repr += ' T[T'
            repr += ' ' + arg_node.indexable_repr()
            repr += ' T]T'
        repr += " ENDARGS"
        return repr

    def get_children(self) -> Generator[AstNode, None, None]:
        return (self.arg_name_to_node[arg.name] for arg in self.implementation.children)

    def as_childless_node(self) -> ChildlessObjectNode:
        choices = tuple([self.arg_name_to_node[arg.name].get_chosen_impl_name()
                         for arg in self.implementation.children])
        return ChildlessObjectNode(self.implementation, choices)


class AstSet:
    def __init__(self):
        self._is_frozen = False

    @abstractmethod
    def freeze(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self.__eq__(other)

    @abstractmethod
    def __hash__(self):
        if self._is_frozen:
            raise ValueError("Unable to hash non-frozen AstSet")

    @abstractmethod
    def is_node_known_valid(self, node: AstNode) -> bool:
        pass


@attr.s(auto_attribs=True)
class ArgsSetData:
    arg_to_choice_set: Mapping[str, 'AstObjectChoiceSet']
    _max_probability: float
    _is_known_valid: bool
    _max_weight: float
    _is_frozen: bool = attr.ib(init=False, default=False)

    def freeze(self):
        self._is_frozen = True
        for node in self.arg_to_choice_set.values():
            node.freeze()

    def __hash__(self):
        if not self._is_frozen:
            raise ValueError("Object must be frozen to be hashed")
        return hash((self.arg_to_choice_set, self._max_probability,
                     self._is_known_valid, self._max_weight))

    @staticmethod
    def create_arg_map(implementation: typecontext.AInixObject) \
        -> Mapping[str, 'AstObjectChoiceSet']:
        return pmap({
            arg.name: AstObjectChoiceSet(arg.type) if arg.type else None
            for arg in implementation.children
        })

    def add_from_other_data(self, node: ObjectNode, new_probability,
                            new_is_known_valid_, new_weight):
        if self._is_frozen:
            raise ValueError("Can't mutate frozen data")
        for arg in node.implementation.children:
            self.arg_to_choice_set[arg.name].add(
                node.arg_name_to_node[arg.name], new_probability,
                new_is_known_valid_, new_weight)
        self._max_probability = max(self._max_probability, new_probability)
        self._is_known_valid = self._is_known_valid or new_is_known_valid_
        self._max_weight = max(self._max_weight, new_weight)


class ObjectNodeSet(AstSet):
    def __init__(self, implementation: typecontext.AInixObject):
        super().__init__()
        self._implementation = implementation
        self.data: MutableMapping[ChildlessObjectNode, ArgsSetData] = {}

    def freeze(self):
        self._is_frozen = True
        for d in self.data.values():
            d.freeze()
        self.data = pmap(self.data)

    def _verify_impl_of_new_node(self, node):
        if node.implementation != self._implementation:
            raise ValueError(
                f"Cannot add node with implementation {node.implementation} into"
                f" set of {self._implementation} objects")

    def add(self, node: ObjectNode, probability: float, is_known_valid: bool, weight: float):
        self._verify_impl_of_new_node(node)
        childless_args = node.as_childless_node()
        if childless_args not in self.data:
            self.data[childless_args] = ArgsSetData(
                ArgsSetData.create_arg_map(node.implementation),
                max_probability=probability,
                is_known_valid=is_known_valid,
                max_weight=weight
            )
        self.data[childless_args].add_from_other_data(
            node, probability, is_known_valid, weight)

    def is_node_known_valid(self, node: ObjectNode) -> bool:
        print("check object ", self._implementation.name)
        childless_args = node.as_childless_node()
        if childless_args not in self.data:
            return False
        node_data = self.data[childless_args]
        if not node_data._is_known_valid:
            return False
        for arg_name, child_node in node.arg_name_to_node.items():
            if not node_data.arg_to_choice_set[arg_name].is_node_known_valid(child_node):
                return False
        return True

    def __hash__(self):
        if not self._is_frozen:
            raise ValueError("Cannot hash non-frozen set")
        return hash((self._implementation, self.data))

    def __eq__(self, other):
        return self._implementation == other._implementation and \
               self.data == other.data


@attr.s(auto_attribs=True, frozen=True)
class ImplementationData:
    arg_data: ObjectNodeSet
    max_probability_valid: float
    known_as_valid: bool
    max_weight: float


class AstObjectChoiceSet(AstSet):
    def __init__(self, type_to_choose: typecontext.AInixType):
        super().__init__()
        self._type_to_choice = type_to_choose
        self._impl_name_to_data: MutableMapping[str, 'ImplementationData'] = {}
        self._max_weight = 0
        self._hash_cache = None

    def freeze(self):
        self._is_frozen = True
        for n in self._impl_name_to_data.values():
            n.arg_data.freeze()
        self._impl_name_to_data = pmap(self._impl_name_to_data)

    def add(
        self,
        node: ObjectChoiceLikeNode,
        probability_valid: float,
        known_as_valid: float,
        weight: float
    ):
        if self._is_frozen:
            raise ValueError("Cannot add to frozen AstObjectChoiceSet")
        existing_data = self._impl_name_to_data.get(node.get_chosen_impl_name(), None)
        if existing_data:
            weight = max(weight, existing_data.max_weight)
            probability_valid = max(probability_valid, existing_data.max_probability_valid)
            known_as_valid = known_as_valid or existing_data.known_as_valid
        # figure out the data's next node
        if existing_data and existing_data.arg_data is not None:
            next_node = existing_data.arg_data
        elif node.chosen_object_node:
            next_node = ObjectNodeSet(node.chosen_object_node.implementation)
        else:
            next_node = None
        new_data = ImplementationData(next_node, probability_valid, known_as_valid, weight)
        print("for type", self._type_to_choice.name, " add ", node.get_chosen_impl_name())
        self._impl_name_to_data[node.get_chosen_impl_name()] = new_data
        if next_node:
            next_node.add(node.chosen_object_node, probability_valid, known_as_valid, weight)

    def is_node_known_valid(self, node: ObjectChoiceLikeNode) -> bool:
        print("type ", self._type_to_choice.name)
        if node.get_chosen_impl_name() not in self._impl_name_to_data:
            print("fail ", node.get_chosen_impl_name())
            return False
        data = self._impl_name_to_data[node.get_chosen_impl_name()]
        if not data.known_as_valid:
            return False
        if data.arg_data is None:
            return node.chosen_object_node is None
        return data.arg_data.is_node_known_valid(node.chosen_object_node)

    def __eq__(self, other):
        return self._type_to_choice == other._type_to_choice and \
               self._impl_name_to_data == other._impl_name_to_data and \
               self._max_weight == other._max_weight

    def __hash__(self):
        if self._hash_cache:
            return self._hash_cache
        super().__hash__()
        hash_val = hash((self._type_to_choice, self._impl_name_to_data))
        self._hash_cache = hash_val
        return hash_val


class StringParser:
    def __init__(
        self,
        type_context: typecontext.TypeContext
    ):
        self._type_context = type_context

    def _make_node_for_arg(
        self,
        arg: typecontext.AInixArgument,
        arg_data: parse_primitives.ObjectParseArgData
    ) -> ObjectChoiceLikeNode:
        """Given data gotten from an ObjectParserResult argument
        and returns the appropriate node"""
        if arg.type is None or arg_data is None:
            next_obj_choice = None
        else:
            next_obj_choice = self._parse_object_choice_node(
                arg_data.slice_string, arg.type_parser, arg.type)

        if arg.required:
            return next_obj_choice
        else:
            is_present = arg_data is not None
            return ArgPresentChoiceNode(arg, is_present, next_obj_choice)

    def _parse_object_node(
        self,
        implementation: typecontext.AInixObject,
        string: str,
        parser: parse_primitives.ObjectParser,
    ) -> ObjectNode:
        """Parses a string into a ObjectNode"""
        object_parse = parser.parse_string(string, implementation)
        arg_name_to_node: Dict[str, ObjectChoiceLikeNode] = {}
        for arg in implementation.children:
            arg_present_data = object_parse.get_arg_present(arg.name)
            arg_name_to_node[arg.name] = self._make_node_for_arg(arg, arg_present_data)
        return ObjectNode(implementation, pmap(arg_name_to_node))

    def _parse_object_choice_node(
        self,
        string: str,
        parser: parse_primitives.TypeParser,
        type: typecontext.AInixType
    ) -> ObjectChoiceNode:
        """Parses a string into a ObjectChoiceNode"""
        result = parser.parse_string(string, type)
        next_object_node = self._parse_object_node(
            result.get_implementation(),  result.get_next_string(), result.next_parser
        )
        return ObjectChoiceNode(type, next_object_node)

    def _get_parser(
        self,
        type_name: str,
        parser_name: Optional[str]
    ) -> parse_primitives.TypeParser:
        if parser_name:
            return self._type_context.get_type_parser_by_name(parser_name)
        else:
            type_instance = self._type_context.get_type_by_name(type_name)
            if type_instance.default_type_parser is None:
                raise ValueError(f"No default type parser available for {type_instance}")
            return type_instance.default_type_parser

    def create_parse_tree(
        self,
        string: str,
        root_type_name: str,
        root_parser_name: str = None
    ) -> ObjectChoiceNode:
        root_parser = self._get_parser(root_type_name, root_parser_name)
        root_type = self._type_context.get_type_by_name(root_type_name)
        return self._parse_object_choice_node(
            string, root_parser, root_type)
