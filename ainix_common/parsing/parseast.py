import parse_primitives
import typecontext
from typing import List, Dict, Optional, Type
from collections import namedtuple
from attr import attrs, attrib


class AstNode:
    def __init__(self, parent: Optional['AstNode']):
        self.parent = parent

    def dump_str(self, indent=0):
        return "  " * indent + str(self) + "\n"


class ObjectChoiceNode(AstNode):
    @attrs
    class _Choice:
        object_node = attrib()
        weight = attrib(default=0.0)

    def __init__(self, type_to_choose: typecontext.AInixType, parent: Optional[AstNode]):
        super(ObjectChoiceNode, self).__init__(parent)
        self.type_to_choose = type_to_choose
        self._normalized = False
        self._valid_choice = namedtuple("ValidChoice", ["object_node", "weight"])
        self._valid_choices: Dict[str, ObjectChoiceNode._Choice] = {}

    def add_valid_choice(
        self,
        implementation: typecontext.AInixObject,
        additional_weight: float
    ) -> 'ObjectNode':
        if implementation.type != self.type_to_choose:
            raise ValueError("Add unexpected choice as valid. Expected type " +
                             self.type_to_choose.name + " got " +
                             implementation.name)
        if implementation.name not in self._valid_choices:
            new_object_node = ObjectNode(implementation, self)
            self._valid_choices[implementation.name] = \
                self._Choice(object_node=new_object_node)
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


class ArgPresentChoiceNode(AstNode):
    def __init__(self, argument: typecontext.AInixArgument, parent: AstNode):
        super(ArgPresentChoiceNode, self).__init__(parent)
        self.argument = argument
        self.is_present = False

    def set_choice(self, is_present: bool):
        # TODO (DNGros): add support for weight for and against being present
        self.is_present = is_present

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


class ObjectNode(AstNode):
    def __init__(self, implementation: typecontext.AInixObject, parent: Optional[AstNode]):
        super(ObjectNode, self).__init__(parent)
        self.implementation = implementation
        self.arg_present_choices = {
            arg.name: ArgPresentChoiceNode(arg, self)
            for arg in implementation.children if not arg.required
        }
        self.next_type_choices = {}
        # Create a direct_sibling data if needed
        self.sibling_present_node = None
        self.sibling_obj_choice_node = None
        if implementation.direct_sibling:
            if not implementation.direct_sibling.required:
                self.sibling_present_node = \
                    ArgPresentChoiceNode(implementation.direct_sibling, self)

    def set_arg_present(self, arg: typecontext.AInixArgument) -> Optional[ObjectChoiceNode]:
        if arg.name in self.arg_present_choices:
            self.arg_present_choices[arg.name].set_choice(True)
        if arg.type is not None:
            new_obj_choice_node = ObjectChoiceNode(arg.type, self)
            self.next_type_choices[arg.name] = new_obj_choice_node
            return new_obj_choice_node
        else:
            return None

    def set_sibling_present(self):
        if self.sibling_present_node:
            self.sibling_present_node.set_choice(True)
        if self.implementation.direct_sibling.type is not None:
            self.sibling_obj_choice_node = \
                ObjectChoiceNode(self.implementation.direct_sibling.type, self)
        return self.sibling_obj_choice_node

    def __eq__(self, other):
        return self.implementation == other.implementation and \
            self.arg_present_choices == other.arg_present_choices and \
            self.next_type_choices == other.next_type_choices and \
            self.sibling_present_node == other.sibling_present_node and \
            self.sibling_obj_choice_node == other.sibling_obj_choice_node

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "<ObjectNode for " + self.implementation.name + ">"

    def dump_str(self, indent=0):
        indent_str = "  " * indent
        s = indent_str + "<ObjectNode obj " + self.implementation.name + "> {\n"
        s += indent_str + "  arg_present_choices: {\n"
        for name, choice in self.arg_present_choices.items():
            s += choice.dump_str(indent + 2)
        s += indent_str + "  }\n"
        s += indent_str + "  next_type_choices: {\n"
        for name, choice in self.next_type_choices.items():
            s += choice.dump_str(indent + 2)
        s += indent_str + "  }\n"
        if self.sibling_present_node:
            s += indent_str + "  sibling_present_node: {\n"
            s += self.sibling_present_node.dump_str(indent + 2)
            s += indent_str + "  }\n"
        if self.sibling_obj_choice_node:
            s += indent_str + "  sibling_obj_choice_node: {\n"
            s += self.sibling_obj_choice_node.dump_str(indent + 2)
            s += indent_str + "  }\n"
        s += indent_str + "}\n"
        return s


class StringParser:
    def __init__(
        self,
        root_type: typecontext.AInixType,
        root_parser: Optional[Type[parse_primitives.TypeParser]] = None
    ):
        self._root_type = root_type
        if root_parser is None:
            if root_type.default_type_parser is None:
                raise ValueError("No default parser available for ", root_type)
            self._root_parser = root_type.default_type_parser
        else:
            self._root_parser = root_parser(root_type)

    @staticmethod
    def _parse_object_choice(
        string: str,
        choice_node: ObjectChoiceNode,
        parser_to_use: parse_primitives.TypeParser,
        preference_weight: float
    ) -> List:
        result: parse_primitives.TypeParserResult = parser_to_use.parse_string(string)
        next_object = result.get_implementation()
        next_object_node = choice_node.add_valid_choice(
            next_object, preference_weight)
        next_parser = result.next_parser
        if next_parser is None:
            raise parse_primitives.AInixParseError(
                "No provided object_name parser for parsed object_name",
                next_object)
        object_parse: parse_primitives.ObjectParserResult = \
            result.next_parser.parse_string(result.get_next_string())

        new_data_for_parse_stack = []
        # Loop through children and add nodes for each that is present
        for arg in next_object.children:
            arg_present_data = object_parse.get_arg_present(arg.name)
            if arg_present_data:
                next_type_choice = next_object_node.set_arg_present(arg)
                if next_type_choice is not None:
                    new_data_for_parse_stack.append(
                        (arg_present_data.slice_string, next_type_choice, arg.type_parser))
        # Figure out if need to parse direct sibling
        sibling_arg_data = object_parse.get_sibling_arg()
        if sibling_arg_data:
            next_type_choice = next_object_node.set_sibling_present()
            if next_type_choice:
                new_data_for_parse_stack.append(
                    (sibling_arg_data.slice_string,
                     next_type_choice,
                     next_object.direct_sibling.value_parser))

        return new_data_for_parse_stack

    def _extend_parse_tree(
            self,
            string: str,
            existing_tree: ObjectChoiceNode,
            preference_weight: float
    ):
        if existing_tree.type_to_choose != self._root_type:
            raise ValueError("Tree parser is extending must root_type of the parser")

        parse_stack = [(string, existing_tree, self._root_parser)]
        while parse_stack:
            cur_string, cur_node, cur_parser = parse_stack.pop()
            new_nodes_to_parse = StringParser._parse_object_choice(
                cur_string, cur_node, cur_parser, preference_weight)
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
