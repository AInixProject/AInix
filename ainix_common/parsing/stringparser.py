from typing import Tuple, Optional, Dict
import attr
from pyrsistent import pmap
import ainix_common.parsing
from ainix_common.parsing import typecontext
from ainix_common.parsing.ast_components import ObjectChoiceNode, ObjectNode
from ainix_common.parsing.parse_primitives import TypeParser, ArgParseDelegation, \
    ParseDelegationReturnMetadata, AInixParseError, TypeParserResult, ImplementationParseDelegation


class StringParser:
    """A string parser is a class which handles the raw handling of a string
    into the AST form. It relies on the different parsers defined in the
    `parse_primitives` module to do parsing of different components of the
    AST. However, this is the main driver of the parsing.

    Part of the responsibility of driving the parsing is handling what is
    referred to as  "delegations" of parsing. While parsing certain objects, a
    ObjectParser might decide it would like to know the result of trying to
    parse a string into one of its arguments. It can then yield a delegation
    of parsing the arg. This StringParser class handles recieving those
    delegations, parsing as needed, and then sending result back into the parser.
    Essentually think of delegations as the recursive calls of a recursive decent
    parser which one parser might need to check the results of the recursion
    in deciding how to parse a string.

    Args:
        type_contest: The context we are parsing in.
    """
    def __init__(
        self,
        type_context: typecontext.TypeContext
    ):
        self._type_context = type_context

    def create_parse_tree(
        self,
        string: str,
        root_type_name: str,
        root_parser_name: str = None
    ) -> ObjectChoiceNode:
        """Converts a string into a AST tree

        Args:
            string: The string we wish to convert into an AST
            root_type_name: The name of the type which is the first type me must
                choose on at the root of the tree.
            root_parser_name: An optional TypeParser name which overrides the
                default parser for the root_type_name.
        """
        root_parser = self._get_root_parser(root_type_name, root_parser_name)
        if not root_parser:
            raise ValueError(f"Unable to get a parser type {root_type_name} and "
                             f"root_parser {root_parser_name}")
        root_type = self._type_context.get_type_by_name(root_type_name)
        new_node, string_metadata = self._parse_object_choice_node(
            string, root_parser, root_type)
        return new_node

    def _parse_object_node(
        self,
        implementation: typecontext.AInixObject,
        string: str,
        parser: ainix_common.parsing.parse_primitives.ObjectParser,
    ) -> Tuple[ObjectNode, ParseDelegationReturnMetadata]:
        """Parses a string into a ObjectNode"""
        object_parse, delegation_to_node_map = self._run_object_parser_with_delegations(
            string, implementation, parser)
        print("Parse return ", object_parse.remaining_start_i)
        arg_name_to_node: Dict[str, ObjectChoiceNode] = {}
        for arg in implementation.children:
            arg_present_data = object_parse.get_arg_present(arg.name)
            arg_name_to_node[arg.name], arg_metadata = \
                self._make_node_for_arg(arg, arg_present_data, delegation_to_node_map)
        my_return_metadata = ParseDelegationReturnMetadata(
            True, string, None, implementation, object_parse.remaining_start_i)
        print("Parse metadata ", my_return_metadata)
        return ObjectNode(implementation, pmap(arg_name_to_node)), my_return_metadata

    def _delegate_object_arg_parse(
        self,
        delegation: ArgParseDelegation
    ) -> Tuple[ParseDelegationReturnMetadata, Optional[ObjectChoiceNode]]:
        """Called to parse an argument which a parser asked to delegate and get
        the resulting remaining string from.

        Returns:
            ParseDelegationReturnMetadata: The return value to send back through into
                the calling parser.
            ObjectChoiceNode: The parsed value for arg. If it was failure, then
                will be None.
        """
        arg = delegation.arg

        # Do parsing if needed.
        if arg.type is not None:
            try:
                arg_type_choice, parse_metadata = self._parse_object_choice_node(
                    delegation.string_to_parse, arg.type_parser, arg.type)
                print("delegated return:", parse_metadata)
                out_delegation_return = parse_metadata.change_what_parsed(arg)
            except AInixParseError as e:
                # TODO (DNGros): Use the metadata rather than exceptions to manage this
                metadata = ParseDelegationReturnMetadata(
                    False, delegation.string_to_parse, delegation.slice_to_parse[0],
                    arg, None, str(e))
                return metadata, None
        else:
            # If has None type, then we don't have to any parsing. Assume it was
            # a success and that the arg is present.
            arg_type_choice = None
            out_delegation_return = ParseDelegationReturnMetadata.make_for_unparsed_string(
                delegation.string_to_parse, arg)

        # Figure out the actual node we need to output
        if not arg.required:
            # If it is an optional node, wrap it in a "is present" present node.
            if arg.type is None:
                object_choice = ObjectNode(arg.is_present_object, pmap({}))
            else:
                parsed_v_as_arg = pmap({
                    typecontext.OPTIONAL_ARGUMENT_NEXT_ARG_NAME: arg_type_choice
                })
                object_choice = ObjectNode(arg.is_present_object, parsed_v_as_arg)
            out_node = ObjectChoiceNode(arg.present_choice_type, object_choice)
        else:
            out_node = arg_type_choice
        return out_delegation_return, out_node

    def _make_node_for_arg(
        self,
        arg: typecontext.AInixArgument,
        arg_data: ainix_common.parsing.parse_primitives.ObjectParseArgData,
        delegation_to_node_map: Dict[ParseDelegationReturnMetadata, ObjectChoiceNode]
    ) -> Tuple[ObjectChoiceNode, Optional[ParseDelegationReturnMetadata]]:
        """
        After running running an Object parsers we get results back for each arg.
        This handles creating nodes for a specific arg results.

        Args:
            arg: The arg we just parsed
            arg_data: The data we got back as a result of the parse
            delegation_to_node_map: A map going from any delegations that already
                have been done to their results.

        Returns:
            new_node: A new object node that represents the data of the arg
            child_metadata: stringparse metadata gotten while parsing the new
                node. If the arg is not present, then it will be None.
        """
        arg_is_present = arg_data is not None
        arg_has_already_been_delegated = arg_is_present and arg_data.set_from_delegation is not None
        if arg_has_already_been_delegated:
            done_delegation = arg_data.set_from_delegation
            return delegation_to_node_map[done_delegation], done_delegation

        if not arg.required:
            arg_string_metadata = None
            if arg_is_present:
                arg_has_substructure_to_parse = arg.type_name is not None
                if arg_has_substructure_to_parse:
                    inner_arg_node, arg_string_metadata = self._parse_object_choice_node(
                        arg_data.slice_string, arg.type_parser, arg.type)
                    arg_map = pmap({typecontext.OPTIONAL_ARGUMENT_NEXT_ARG_NAME: inner_arg_node})
                else:
                    arg_map = pmap({})
                object_choice = ObjectNode(arg.is_present_object, arg_map)
            else:
                object_choice = ObjectNode(arg.not_present_object, pmap({}))
            return ObjectChoiceNode(arg.present_choice_type, object_choice), arg_string_metadata
        else:
            return self._parse_object_choice_node(
                arg_data.slice_string, arg.type_parser, arg.type
            )

    def _run_object_parser_with_delegations(
        self,
        string: str,
        implementation: typecontext.AInixObject,
        parser: ainix_common.parsing.parse_primitives.ObjectParser
    ) -> Tuple[
        ainix_common.parsing.parse_primitives.ObjectParserResult,
        Dict[ParseDelegationReturnMetadata, ObjectChoiceNode]
    ]:
        """Will run a specified object parser. If the parser asks to delegate
        the parsing of any its parsing, it will handle that and pass back the
        results."""
        parser_gen = parser.parse_string(string, implementation)
        delegation_to_node: Dict[ParseDelegationReturnMetadata, ObjectChoiceNode] = {}
        last_delegation_result = None
        while True:
            try:
                delegation = parser_gen.send(last_delegation_result)
            except StopIteration as stop_iter:
                return_result = stop_iter.value
                return return_result, delegation_to_node
            # Well it didn't reach the end, so it must have yielded a delegation. Handle that.
            delegation_return, out_node = self._delegate_object_arg_parse(delegation)
            print(delegation_return)
            delegation_to_node[delegation_return] = out_node
            last_delegation_result = delegation_return

    def _object_choice_result_to_string_metadata(
        self,
        result: ainix_common.parsing.parse_primitives.TypeParserResult,
        child_metadata: ParseDelegationReturnMetadata
    ):
        """Converts the result we get from a type parser into a string metadata
        result."""
        si, endi = result.get_next_slice()
        si += child_metadata.remaining_right_starti
        endi += child_metadata.remaining_right_starti
        return ParseDelegationReturnMetadata(True, result.string, 0, result.type, si)

    def _parse_object_choice_node(
        self,
        string: str,
        parser: TypeParser,
        type: typecontext.AInixType
    ) -> Tuple[ObjectChoiceNode, ParseDelegationReturnMetadata]:
        """Parses a string into a ObjectChoiceNode. This is more internal use.
        For the more user friendly method see create_parse_tree()"""
        result, delegation_map = self._run_type_parser_with_delegations(string, type, parser)
        if result._accepted_delegation is not None:
            next_object_node, child_string_metadata = delegation_map[result._accepted_delegation]
        else:
            next_object_node, child_string_metadata = self._parse_object_node(
                result.get_implementation(),  result.get_next_string(), result.next_parser
            )
        metadata = self._object_choice_result_to_string_metadata(result, child_string_metadata)
        print("parse object choice", metadata)
        return ObjectChoiceNode(type, next_object_node), metadata

    def _run_type_parser_with_delegations(
        self,
        string: str,
        type_to_parser: typecontext.AInixType,
        parser: ainix_common.parsing.parse_primitives.TypeParser
    ) -> Tuple[TypeParserResult, Dict[ParseDelegationReturnMetadata, ObjectNode]]:
        """Run a type parser handling delegations it yields as needed."""
        parser_gen = parser.parse_string(string, type_to_parser)
        delegation_to_node: Dict[ParseDelegationReturnMetadata, ObjectNode] = {}
        last_delegation_result = None
        while True:
            try:
                delegation = parser_gen.send(last_delegation_result)
            except StopIteration as stop_iter:
                return_result = stop_iter.value
                print("run type parser delegations return", return_result)
                return return_result, delegation_to_node
            # Well it didn't reach the end, so it must have yielded a delegation. Handle that.
            node, return_metadata = self._delegate_object_arg_parse(delegation)
            delegation_to_node[return_metadata] = node
            last_delegation_result = return_metadata

    def _delegate_impl_parser(self, delegation: ImplementationParseDelegation):
        try:
            node, return_metadata = self._parse_object_node(
                delegation.implementation,
                delegation.string_to_parse,
                delegation.next_parser
            )
            return node, return_metadata
        except AInixParseError as e:
            fail_return = ParseDelegationReturnMetadata(
                False, delegation.string_to_parse, None, delegation.implementation, None, str(e))
            return None, fail_return

    def _get_root_parser(
        self,
        type_name: str,
        parser_name: Optional[str]
    ) -> ainix_common.parsing.parse_primitives.TypeParser:
        """The public interface accepts a type name to parse as the root AST node
        for the string. This method converts that type name into an actuall parser
        instance."""
        if parser_name:
            return self._type_context.get_type_parser_by_name(parser_name)
        else:
            type_instance = self._type_context.get_type_by_name(type_name)
            if type_instance.default_type_parser is None:
                raise ValueError(f"No default type parser available for {type_instance}")
            return type_instance.default_type_parser
