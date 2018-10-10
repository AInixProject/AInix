from typing import Tuple, Optional, Dict
import attr
from pyrsistent import pmap
import ainix_common.parsing
from ainix_common.parsing import typecontext
from ainix_common.parsing.ast_components import ObjectChoiceNode, ObjectNode
from ainix_common.parsing.parse_primitives import TypeParser


@attr.s(auto_attribs=True, frozen=True)
class StringParseResultMetadata:
    """
    Keeps track of metadata about a result. Right now this just stores data
    for use in a LL style parse. However, in the future it might store a
    tree linking each part of the parse tree with a substring that it was
    derived from. That will be implemented as needed.
    Args:
        string: The string which was origionally parsed
        remaining_right_starti: When calling a either parse_object_node or
            parse_object_choice_node the parser will recurse down parsing
            all possible substructures of the parse. While doing this we
            keep track the farthest right part of the string which was
            not used during parsing. This variable stores the index into the
            string which is the start of that far rightmost part of unused
            string characters. This can be used for implementing a LL style
            parse.
        remaining_right_endi: Type parsers are able to return an arbitrary
            next string slice. This might not cut off some to the right.
            If that is the case this returns the end index (exclusive), of the
            rightmost remaining (non-consumed) string.
            NOTE: We are not really sure what this would be actually used for.
            If a type parser was operating a LL style it shouldn't really
            do this. However, it could. This might be useful somehow, or might
            just need to be removed and not supported.
    """
    string: str
    remaining_right_starti: int
    #remaining_right_endi: int

    @staticmethod
    def make_for_unparsed_string(string: str) -> 'StringParseResultMetadata':
        """A metadata with nothing consumed yet"""
        return StringParseResultMetadata(string, 0)

    def combine_with_child_metadata(
        self,
        child_meta_data: 'StringParseResultMetadata',
        child_start_offset: int
    ) -> 'StringParseResultMetadata':
        new_remaining_starti = max(self.remaining_right_starti,
                                   child_meta_data.remaining_right_starti + child_start_offset)
        #new_remaining_endi = min(self.remaining_right_endi,
        #                           child_meta_data.remaining_right_endi + child_start_offset)
        return StringParseResultMetadata(self.string, new_remaining_starti)


class StringParser:
    def __init__(
        self,
        type_context: typecontext.TypeContext
    ):
        self._type_context = type_context

    def _make_node_for_arg(
        self,
        arg: typecontext.AInixArgument,
        arg_data: ainix_common.parsing.parse_primitives.ObjectParseArgData
    ) -> Tuple[ObjectChoiceNode, Optional[StringParseResultMetadata]]:
        """
        Args:
            arg: The arg we just parsed
            arg_data: The data we got back as a result of the parse

        Returns:
            new_node: A new object node that represents the data of the arg
            child_metadata: stringparse metadata gotten while parsing the new
                node. If the arg is not present, then it will be None.
        """
        if not arg.required:
            arg_string_metadata = None
            arg_is_present = arg_data is not None
            if arg_is_present:
                arg_has_substructure_to_parse = arg.type_name is not None
                if arg_has_substructure_to_parse:
                    inner_arg_node, arg_string_metadata = self.parse_object_choice_node(
                        arg_data.slice_string, arg.type_parser, arg.type)
                    arg_map = pmap({typecontext.OPTIONAL_ARGUMENT_NEXT_ARG_NAME: inner_arg_node})
                else:
                    arg_map = pmap({})
                object_choice = ObjectNode(arg.is_present_object, arg_map)
            else:
                object_choice = ObjectNode(arg.not_present_object, pmap({}))
            return ObjectChoiceNode(arg.present_choice_type, object_choice), arg_string_metadata
        else:
            return self.parse_object_choice_node(
                arg_data.slice_string, arg.type_parser, arg.type
            )

    def parse_object_node(
        self,
        implementation: typecontext.AInixObject,
        string: str,
        parser: ainix_common.parsing.parse_primitives.ObjectParser,
    ) -> Tuple[ObjectNode, StringParseResultMetadata]:
        """Parses a string into a ObjectNode"""
        object_parse = parser.parse_string(string, implementation)
        arg_name_to_node: Dict[str, ObjectChoiceNode] = {}
        my_metadata = StringParseResultMetadata.make_for_unparsed_string(string)
        for arg in implementation.children:
            arg_present_data = object_parse.get_arg_present(arg.name)
            arg_name_to_node[arg.name], arg_metadata = \
                self._make_node_for_arg(arg, arg_present_data)
            if arg_metadata:
                start_of_arg_substring, _ = arg_present_data.slice
                my_metadata.combine_with_child_metadata(my_metadata, start_of_arg_substring)
        return ObjectNode(implementation, pmap(arg_name_to_node)), my_metadata

    def _object_choice_result_to_string_metadata(self, result: ainix_common.parsing.parse_primitives.TypeParserResult):
        """Converts the result we get from a type parser into a string metadata
        result."""
        si, endi = result.get_next_slice()
        return StringParseResultMetadata(result.string, endi)

    def parse_object_choice_node(
        self,
        string: str,
        parser: TypeParser,
        type: typecontext.AInixType
    ) -> Tuple[ObjectChoiceNode, StringParseResultMetadata]:
        """Parses a string into a ObjectChoiceNode. This is more internal use.
        For the more user friendly method see create_parse_tree()"""
        result = parser.parse_string(string, type)
        next_object_node, child_string_metadata = self.parse_object_node(
            result.get_implementation(),  result.get_next_string(), result.next_parser
        )
        metadata = self._object_choice_result_to_string_metadata(result)
        return ObjectChoiceNode(type, next_object_node), metadata

    def _get_parser(
        self,
        type_name: str,
        parser_name: Optional[str]
    ) -> ainix_common.parsing.parse_primitives.TypeParser:
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
        """Converts a string into a AST tree"""
        root_parser = self._get_parser(root_type_name, root_parser_name)
        if not root_parser:
            raise ValueError(f"Unable to get a parser type {root_type_name} and "
                             f"root_parser {root_parser_name}")
        root_type = self._type_context.get_type_by_name(root_type_name)
        new_node, string_metadata = self.parse_object_choice_node(
            string, root_parser, root_type)
        return new_node