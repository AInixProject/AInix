import parse_primitives
import typecontext
from typing import List, Dict, Optional, Type, Union
from attr import attrs, attrib
from abc import ABC, abstractmethod


def indexable_repr_classify_type(type_name: str):
    return f"CLASSIFY_TYPE={type_name}"


def indexable_repr_object(object_name: str):
    return f"OBJECT={object_name}"


def convert_ast_to_indexable_repr():
    pass


class AstNode(ABC):
    def __init__(self, parent: Optional['AstNode']):
        self.parent = parent

    def dump_str(self, indent=0):
        return "  " * indent + str(self) + "\n"

    @abstractmethod
    def get_children(self) -> List['AstNode']:
        """Returns all descendants of this node"""
        pass


class ObjectChoiceNode(AstNode):
    @attrs(auto_attribs=True)
    class _Choice:
        object_node: 'ObjectNode'
        weight: float = 1

    def __init__(self, type_to_choose: typecontext.AInixType, parent: Optional[AstNode]):
        super(ObjectChoiceNode, self).__init__(parent)
        self.type_to_choose = type_to_choose
        self._normalized = False
        self._valid_choices: Dict[str, ObjectChoiceNode._Choice] = {}

    # TODO (DNGros): seporate out a single AST and multiple valid AST holder
    def add_valid_choice(
        self,
        implementation: typecontext.AInixObject,
        additional_weight: float
    ) -> 'ObjectNode':
        if implementation.type != self.type_to_choose:
            raise ValueError("Add unexpected choice as valid. Expected type " +
                             self.type_to_choose.name + " got " +
                             implementation.type_name)
        if implementation.name not in self._valid_choices:
            new_object_node = ObjectNode(implementation, self)
            choice = self._valid_choices[implementation.name] = \
                self._Choice(object_node=new_object_node,
                             weight=additional_weight)
        else:
            choice = self._valid_choices[implementation.name]
            choice.weight += additional_weight
        return choice.object_node

    def __eq__(self, other):
        # Note we don't actually check parent because that
        # would be recursive loop.
        if not isinstance(other, ObjectChoiceNode):
            return False
        return self.type_to_choose == other.type_to_choose and \
            self._valid_choices == other._valid_choices

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        s = "<ObjectChoiceNode for " + str(self.type_to_choose.name)
        s += " valid_choices=" + str(self._valid_choices)
        s += ">"
        return s

    def dump_str(self, indent=0):
        indent_str = "  " * indent
        s = indent_str + "<ObjectChoiceNode type " + self.type_to_choose.name + "> {\n"
        for name, choice in self._valid_choices.items():
            s += indent_str + '  Weight ' + str(choice.weight) + "\n"
            s += choice.object_node.dump_str(indent + 2)
        s += indent_str + "}\n"
        return s

    def indexable_repr(self) -> str:
        repr = indexable_repr_classify_type(self.type_to_choose.name)
        # TODO (DNGros): don't just take the zeroth. Refactor for this AST
        assert len(self._valid_choices) == 1
        choosen = list(self._valid_choices.values())[0]
        repr += f" O[O {choosen.object_node.indexable_repr()} O]O"
        return repr

    def get_children(self) -> List['AstNode']:
        # TODO (DNGros): refactor so only one for this kind of AST
        assert len(self._valid_choices) == 1
        return [list(self._valid_choices.values())[0].object_node]


class ArgPresentChoiceNode(AstNode):
    def __init__(self, argument: typecontext.AInixArgument, parent: AstNode):
        super(ArgPresentChoiceNode, self).__init__(parent)
        self.argument = argument
        self.is_present = False
        self.obj_choice_node = None

    def set_choice(self, is_present: bool) -> Optional[ObjectChoiceNode]:
        # TODO (DNGros): add support for weight for and against being present
        self.is_present = is_present
        if is_present and self.argument.type:
            self.obj_choice_node = ObjectChoiceNode(self.argument.type, self)
        return self.obj_choice_node

    def __eq__(self, other):
        return self.argument == other.argument and \
               self.is_present == other.is_present

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "<ArgPresentChoiceNode for " + str(self.argument) + \
            ". " + str(self.is_present) + ">"

    def dump_str(self, indent=0):
        indent_str = "  " * indent
        s = indent_str + "<ArgPresentChoiceNode for " + self.argument.name +\
            ". " + str(self.is_present) + ">\n"
        return s

    def indexable_repr(self) -> str:
        # Just pretend to be a type.
        # TODO (DNGros): This is very not clean. Optional args should define actual types
        repr = indexable_repr_classify_type(self.argument.present_choice_type_name)
        if self.is_present:
            repr += f" O[O {indexable_repr_object(self.argument.is_present_name)} ARGS"
            if self.obj_choice_node:
                repr += f" ARG={self.argument.is_present_name}::VAL"
                repr += f" T[T {self.obj_choice_node.indexable_rep()} T]T"
            repr += " ENDARGS"
            repr += f" O]O"
        else:
            repr += f" O[O {indexable_repr_object(self.argument.not_present_name)} ARGS"
            repr += " ENDARGS O]O"
        return repr

    def get_children(self) -> List['AstNode']:
        return [self.obj_choice_node] if self.obj_choice_node else []


class ObjectNode(AstNode):
    def __init__(self, implementation: typecontext.AInixObject, parent: Optional[AstNode]):
        super(ObjectNode, self).__init__(parent)
        self.implementation = implementation
        self.arg_name_to_node: Dict[str, Union[ObjectChoiceNode, ArgPresentChoiceNode]] = {}
        self._add_arg_present_choices()

    def _add_arg_present_choices(self):
        for arg in self.implementation.children:
            if not arg.required:
                self.arg_name_to_node[arg.name] = ArgPresentChoiceNode(arg, self)

    def set_arg_present(self, arg: typecontext.AInixArgument) -> Optional[ObjectChoiceNode]:
        if not arg.required:
            already_made_present_choice = self.arg_name_to_node[arg.name]
            return already_made_present_choice.set_choice(True)
        elif arg.type is not None:
            new_obj_choice_node = ObjectChoiceNode(arg.type, self)
            self.arg_name_to_node[arg.name] = new_obj_choice_node
            return new_obj_choice_node
        else:
            return None

    def __eq__(self, other):
        return self.implementation == other.implementation and \
            self.arg_name_to_node == other.arg_name_to_node

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
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

    def get_children(self) -> List[AstNode]:
        return [node for arg_name, node in self.arg_name_to_node.items()]


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

    def _extend_parse_tree(
        self,
        string: str,
        existing_tree: ObjectChoiceNode,
        preference_weight: float
    ):
        if existing_tree.type_to_choose != self._root_type:
            raise ValueError("Tree parser is extending must root_type of the parser")

        parse_stack = [(string, existing_tree, self._root_parser, self._root_type)]
        while parse_stack:
            cur_string, cur_node, cur_parser, cur_type = parse_stack.pop()
            new_nodes_to_parse = StringParser._parse_object_choice(
                cur_string, cur_node, cur_parser, cur_type, preference_weight)
            parse_stack.extend(new_nodes_to_parse)

    def create_parse_tree(self, string: str, preference_weight: float = 1):
        new_tree = ObjectChoiceNode(self._root_type, parent=None)
        self._extend_parse_tree(string, new_tree, preference_weight)
        return new_tree

    def create_parse_tree_multi(
        self,
        string: List[str],
        preference_weight: List[float]
    ):
        raise NotImplementedError("Need to do this")

    def extend_parse_tree(
        self,
        string: str,
        existing_tree: ObjectChoiceNode,
        preference_weight: float = 1
    ):
        self._extend_parse_tree(string, existing_tree, preference_weight)
