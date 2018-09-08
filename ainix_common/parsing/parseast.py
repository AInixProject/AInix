import parse_primitives
import typecontext
from typing import List, Dict, Optional, Type, Union, Generator
from attr import attrs, attrib
from abc import ABC, abstractmethod, abstractproperty
from functools import lru_cache
import attr


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


@attr.s(auto_attribs=True, frozen=True, cache_hash=True)
class ObjectChoiceNode(ObjectChoiceLikeNode):
    type_to_choose: typecontext.AInixType
    choice: 'ObjectNode'

    def __attrs_post_init__(self):
        if self.choice.implementation.type_name != self.type_to_choose.name:
            raise ValueError("Add unexpected choice as valid. Expected type " +
                             self.get_type_to_choose_name() + " got " +
                             self.choice.implementation.type_name)

    def get_type_to_choose_name(self) -> str:
        return  self.type_to_choose.name

    @property
    def type_context(self) -> typecontext.TypeContext:
        return self.type_to_choose.type_context

    def get_chosen_impl_name(self) -> str:
        return self.choice.implementation.name

    def __str__(self):
        s = "<ObjectChoiceNode for " + str(self.get_type_to_choose_name())
        s += " valid_choices=" + str(self.choice)
        s += ">"
        return s

    def dump_str(self, indent=0):
        indent_str = "  " * indent
        s = indent_str + "<ObjectChoiceNode type " + self.get_type_to_choose_name() + "> {\n"
        s += self.choice.dump_str(indent + 2)
        s += indent_str + "}\n"
        return s

    def indexable_repr(self) -> str:
        repr = indexable_repr_classify_type(self.get_type_to_choose_name())
        repr += f" O[O {self.choice.indexable_repr()} O]O"
        return repr

    def get_children(self) -> Generator[AstNode, None, None]:
        assert self.choice is not None
        yield self.choice


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


@attr.s(auto_attribs=True, frozen=True, cache_hash=True)
class ObjectNode(AstNode):
    implementation: typecontext.AInixObject
    arg_name_to_node: Dict[str, ObjectChoiceLikeNode]

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


class StringParser:
    def __init__(
        self,
        root_type: typecontext.AInixType,
        root_parser: Optional[Type[parse_primitives.TypeParser]] = None
    ):
        self._root_type = root_type
        if root_parser is None:
            if root_type.default_type_parser is None:
                raise ValueError(f"No default type parser available for {root_type}")
            self._root_parser = root_type.default_type_parser
        else:
            self._root_parser = root_parser(root_type)

    @staticmethod
    def _parse_object_choice(
        string: str,
        choice_node: ObjectChoiceNode,
        parser_to_use: parse_primitives.TypeParser,
        current_type_to_parse: typecontext.AInixType,
        preference_weight: float
    ) -> List:
        result: parse_primitives.TypeParserResult = \
            parser_to_use.parse_string(string, current_type_to_parse)
        next_object = result.get_implementation()
        next_object_node = choice_node.add_valid_choice(
            next_object, preference_weight)
        next_parser = result.next_parser
        if next_parser is None:
            raise parse_primitives.AInixParseError(
                f"No provided object parser for parsed object {next_object}")
        object_parse: parse_primitives.ObjectParserResult = \
            result.next_parser.parse_string(result.get_next_string(), next_object)

        new_data_for_parse_stack = []
        # Loop through children and add nodes for each that is present
        for arg in next_object.children:
            arg_present_data = object_parse.get_arg_present(arg.name)
            if arg_present_data:
                next_type_choice = next_object_node.set_arg_present(arg)
                if next_type_choice is not None:
                    new_parse_entry = (arg_present_data.slice_string,
                                       next_type_choice,
                                       arg.type_parser, arg.type)
                    new_data_for_parse_stack.append(new_parse_entry)

        return new_data_for_parse_stack

    #def _extend_parse_tree(
    #    self,
    #    string: str,
    #    existing_tree: ObjectChoiceNode,
    #    preference_weight: float
    #):
    #    if existing_tree.get_type_to_choose_name() != self._root_type.name:
    #        raise ValueError("Tree parser is extending must root_type of the parser")

    #    parse_stack = [(string, existing_tree, self._root_parser, self._root_type)]
    #    while parse_stack:
    #        cur_string, cur_node, cur_parser, cur_type = parse_stack.pop()
    #        new_nodes_to_parse = StringParser._parse_object_choice(
    #            cur_string, cur_node, cur_parser, cur_type, preference_weight)
    #        parse_stack.extend(new_nodes_to_parse)
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
        return ObjectNode(implementation, arg_name_to_node)

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

    def create_parse_tree(self, string: str) -> ObjectChoiceNode:
        return self._parse_object_choice_node(
            string, self._root_parser, self._root_type)

    # TODO (DNGros): MAKE AST IMMUTABLE! then cache
    #@lru_cache
    #def create_parse_tree(self, string: str, preference_weight: float = 1):
    #    new_tree = ObjectChoiceNode(self._root_type, parent=None)
    #    self._extend_parse_tree(string, new_tree, preference_weight)
    #    return new_tree

    #def create_parse_tree_multi(
    #    self,
    #    string: List[str],
    #    preference_weight: List[float]
    #):
    #    raise NotImplementedError("Need to do this")

    #def extend_parse_tree(
    #    self,
    #    string: str,
    #    existing_tree: ObjectChoiceNode,
    #    preference_weight: float = 1
    #):
    #    self._extend_parse_tree(string, existing_tree, preference_weight)


class MultitypeStringParser:
    """A string parser that can operate on any type. It does this by cacheing
    individual StringParsers for each type"""
    def __init__(self, type_context: typecontext.TypeContext):
        self._type_context = type_context

    @lru_cache()
    def _get_string_parser_for_type(self, type_name: str):
        type_instance = self._type_context.get_type_by_name(type_name)
        return StringParser(type_instance)

    def create_parse_tree(
        self,
        string: str,
        type_name: str,
        preference_weight: float = 1
    ) -> ObjectChoiceNode:
        parser = self._get_string_parser_for_type(type_name)
        return parser.create_parse_tree(string, preference_weight)
