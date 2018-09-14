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
        yield from map(self.depth_first_iter, self.get_children())

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

#class ObjectChoiceLikeNode(AstNode):
#    @abstractmethod
#    def get_type_to_choose_name(self) -> str:
#        pass
#
#    @property
#    @abstractmethod
#    def type_context(self) -> typecontext.TypeContext:
#        pass
#
#    @abstractmethod
#    def get_chosen_impl_name(self) -> str:
#        pass
#
#    @abstractmethod
#    def indexable_repr(self) -> str:
#        pass
#
#    @property
#    @abstractmethod
#    def next_node(self) -> Optional[AstNode]:
#        pass

class ObjectChoiceNode(AstNode):
    def __init__(
        self,
        type_to_choose: typecontext.AInixType,
        frozen_choice: 'ObjectNode' = None
    ):
        self._type_to_choose = type_to_choose
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
                             self._choice._implementation.type_name)

    def get_type_to_choose_name(self) -> str:
        return  self._type_to_choose.name

    @property
    def next_node(self) -> 'ObjectNode':
        return self._choice

    @property
    def type_context(self) -> typecontext.TypeContext:
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
        assert self._choice is not None
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
        return self._type_to_choose == other._type_to_choose and \
               self._choice == other._choice



#@attr.s(auto_attribs=True, frozen=True, cache_hash=True)
#class ArgPresentChoiceNode(ObjectChoiceLikeNode):
#    argument: typecontext.AInixArgument
#    is_present: bool
#    obj_choice_node: Optional['ObjectChoiceNode']
#
#    def __attrs_post_init__(self):
#        if self.is_present and self.obj_choice_node is None and \
#                self.argument.type is not None:
#            raise ValueError(f"Argument {self.argument.name} requires an value, yet"
#                             f" a ArgPresentChoice node constructed with no"
#                             f"obj_choice_node")
#        if self.obj_choice_node is not None and self.argument.type is None:
#            raise ValueError(f"Argument {self.argument.name} has no type, yet"
#                             f"a ArgPresentChoice node constructed with an"
#                             f"ObjectChoiceNode.")
#
#    def get_type_to_choose_name(self) -> str:
#        return self.argument.present_choice_type_name
#
#    @property
#    def next_node(self) -> Optional['ObjectChoiceLikeNode']:
#        if not self.is_present:
#            return None
#        return self.obj_choice_node
#
#    @property
#    def type_context(self) -> typecontext.TypeContext:
#        return self.argument._type_context
#
#    def get_chosen_impl_name(self) -> str:
#        return self.argument.is_present_name if self.is_present else \
#            self.argument.not_present_name
#
#    #def set_valid_choice_by_name(self, obj_name: str) -> 'AstNode':
#    #    if obj_name == self.argument.is_present_name:
#    #        return self.set_choice(True)
#    #    elif obj_name == self.argument.not_present_name:
#    #        return self.set_choice(False)
#    #    else:
#    #        raise ValueError(f"tried to set ArgPresentChoiceNode with invalid"
#    #                         f" object name {obj_name}")
#
#    def __str__(self):
#        return "<ArgPresentChoiceNode for " + str(self.argument) + \
#            ". " + str(self.is_present) + ">"
#
#    def dump_str(self, indent=0):
#        indent_str = "  " * indent
#        s = indent_str + "<ArgPresentChoiceNode for " + self.argument.name +\
#            ". " + str(self.is_present) + "> {\n"
#        if self.obj_choice_node:
#            s += self.obj_choice_node.dump_str(indent + 1)
#        s += indent_str + "}\n"
#        return s
#
#    def indexable_repr(self) -> str:
#        # Just pretend to be a type in index output
#        # TODO (DNGros): This is very not clean. Optional args should probably define own types
#        repr = indexable_repr_classify_type(self.get_type_to_choose_name())
#        repr += f" O[O {indexable_repr_object(self.get_chosen_impl_name())} ARGS"
#        if self.is_present and self.obj_choice_node:
#            repr += f" ARG={self.argument.is_present_name}::VAL"
#            repr += f" T[T {self.obj_choice_node.indexable_repr()} T]T"
#        repr += " ENDARGS O]O"
#        return repr
#
#    def get_children(self) -> Generator[AstNode, None, None]:
#        if self.obj_choice_node:
#            yield self.obj_choice_node


@attr.s(auto_attribs=True, frozen=True)
class ChildlessObjectNode:
    implementation: typecontext.AInixObject
    chosen_type_names: Tuple[str, ...]


#@attr.s(auto_attribs=True, frozen=True, cache_hash=True)
class ObjectNode(AstNode):
    def __init__(
        self,
        implementation: typecontext.AInixObject,
        frozen_arg_name_to_node: Mapping[str, ObjectChoiceNode] = None
    ):
        self._implementation = implementation
        if frozen_arg_name_to_node:
            self._arg_name_to_node = frozen_arg_name_to_node
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
        return (self._arg_name_to_node[arg.name]
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
        return self._implementation == other._implementation and \
               self._arg_name_to_node == other._arg_name_to_node


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
    def add(self, node: AstNode, known_as_valid: float, weight: float, probability_valid: float) -> None:
        pass


class ArgsSetData:
    def __init__(self, implementation: typecontext.AInixObject, part_of: 'ObjectNodeSet'):
        self.arg_to_choice_set: Mapping[str, 'AstObjectChoiceSet'] = {
            arg.name: AstObjectChoiceSet(arg.type, part_of) if arg.type else None
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

    @staticmethod
    def _create_arg_map(implementation: typecontext.AInixObject) \
        -> Mapping[str, 'AstObjectChoiceSet']:
        return pmap({
        })

    def add_from_other_data(self, node: ObjectNode, new_probability,
                            new_is_known_valid_, new_weight):
        if self._is_frozen:
            raise ValueError("Can't mutate frozen data")
        for arg in node._implementation.children:
            print("add_from_other_data", arg.name)
            self.arg_to_choice_set[arg.name].add(node._arg_name_to_node[arg.name],
                                                 new_is_known_valid_, new_weight, new_probability)
        self._max_probability = max(self._max_probability, new_probability)
        self._is_known_valid = self._is_known_valid or new_is_known_valid_
        self._max_weight = max(self._max_weight, new_weight)


class ObjectNodeSet(AstSet):
    def __init__(self, implementation: typecontext.AInixObject, parent: Optional[AstSet]):
        super().__init__(parent)
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

    def add(self, node: ObjectNode, known_as_valid: bool, weight: float, probability_valid: float):
        self._verify_impl_of_new_node(node)
        childless_args = node.as_childless_node()
        print("childless args", childless_args)
        if childless_args not in self.data:
            print("  creating new v in ObjectNodeSet", known_as_valid)
            self.data[childless_args] = ArgsSetData(node._implementation, self)
        self.data[childless_args].add_from_other_data(
            node, probability_valid, known_as_valid, weight)

    def is_node_known_valid(self, node: ObjectNode) -> bool:
        print("check object ", self._implementation.name)
        print("  args", node.as_childless_node())
        childless_args = node.as_childless_node()
        if childless_args not in self.data:
            return False
        node_data = self.data[childless_args]
        print("  node_data", node_data)
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
    next_node: AstSet
    max_probability_valid: float
    known_as_valid: bool
    max_weight: float


class AstObjectChoiceSet(AstSet):
    def __init__(self, type_to_choose: typecontext.AInixType, parent: Optional[AstSet]):
        super().__init__(parent)
        self._type_to_choice = type_to_choose
        self._impl_name_to_data: MutableMapping[str, 'ImplementationData'] = {}
        self._max_weight = 0
        self._hash_cache = None

    @property
    def type_to_choose_name(self):
        return self._type_to_choice.name

    def freeze(self):
        self._is_frozen = True
        for n in self._impl_name_to_data.values():
            if n.next_node:
                n.next_node.freeze()
        self._impl_name_to_data = pmap(self._impl_name_to_data)

    def add(
        self,
        node: ObjectChoiceNode,
        known_as_valid: float,
        weight: float,
        probability_valid: float
    ):
        if self._is_frozen:
            raise ValueError("Cannot add to frozen AstObjectChoiceSet")
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
        print("for type", self._type_to_choice.name, " add ", node.get_chosen_impl_name())
        self._impl_name_to_data[node.get_chosen_impl_name()] = new_data
        if next_node:
            next_node.add(node.next_node, known_as_valid, weight, probability_valid)

    def is_node_known_valid(self, node: ObjectChoiceNode) -> bool:
        print("type ", self._type_to_choice.name)
        if node.get_chosen_impl_name() not in self._impl_name_to_data:
            print("fail ", node.get_chosen_impl_name())
            return False
        data = self._impl_name_to_data[node.get_chosen_impl_name()]
        if not data.known_as_valid:
            return False
        if data.next_node is None:
            return node.next_node is None
        return data.next_node.is_node_known_valid(node.next_node)

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
    ) -> ObjectChoiceNode:
        """Given data gotten from an ObjectParserResult argument
        and returns the appropriate node"""
        if not arg.required:
            arg_is_present = arg_data is not None
            if arg_is_present:
                if arg.type is not None:
                    inner_arg_node = self._parse_object_choice_node(
                        arg_data.slice_string, arg.type_parser, arg.type)
                    arg_map = pmap({typecontext.OPTIONAL_ARGUMENT_NEXT_ARG_NAME : inner_arg_node})
                else:
                    arg_map = pmap({})
                object_choice = ObjectNode(arg.is_present_object, arg_map)
            else:
                object_choice = ObjectNode(arg.not_present_object, pmap({}))
            return ObjectChoiceNode(arg.present_choice_type, object_choice)
        else:
            return self._parse_object_choice_node(
                arg_data.slice_string, arg.type_parser, arg.type
            )

    def _parse_object_node(
        self,
        implementation: typecontext.AInixObject,
        string: str,
        parser: parse_primitives.ObjectParser,
    ) -> ObjectNode:
        """Parses a string into a ObjectNode"""
        object_parse = parser.parse_string(string, implementation)
        arg_name_to_node: Dict[str, ObjectChoiceNode] = {}
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
        if not root_parser:
            raise ValueError(f"Unable to get a parser type {root_type_name} and "
                             f"root_parser {root_parser_name}")
        root_type = self._type_context.get_type_by_name(root_type_name)
        return self._parse_object_choice_node(
            string, root_parser, root_type)
