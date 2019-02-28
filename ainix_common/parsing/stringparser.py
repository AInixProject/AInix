"""Userfacing classes for converting strings into Abstract Syntax Trees (AST's)
or ASTs into strings"""
from typing import Tuple, Optional, Dict, Union, Mapping
import attr
from pyrsistent import pmap
import ainix_common.parsing
from ainix_common.parsing import typecontext
from ainix_common.parsing.ast_components import ObjectChoiceNode, ObjectNode, CopyNode, \
    ObjectNodeLike, is_obj_choice_a_not_present_node, is_obj_choice_a_present_node, AstIterPointer
from ainix_common.parsing.model_specific.tokenizers import StringTokensMetadata
from ainix_common.parsing.parse_primitives import (TypeParser, ArgParseDelegation,
                                                   ParseDelegationReturnMetadata, AInixParseError,
                                                   TypeParserResult,
                                                   ImplementationParseDelegation,
                                                   TypeToStringResult,
                                                   ImplementationToStringDelegation, ObjectParser,
                                                   ObjectNodeArgMap, ArgToStringDelegation,
                                                   ArgIsPresentToString)
from ainix_common.parsing.typecontext import OPTIONAL_ARGUMENT_NEXT_ARG_NAME
from ainix_common.parsing.model_specific import tokenizers
import functools
import pprint


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

    @functools.lru_cache(maxsize=500)
    def create_parse_tree(
        self,
        string: str,
        root_type_name: str,
        root_parser_name: str = None,
        allow_partial_consume: bool = False
    ) -> ObjectChoiceNode:
        """Converts a string into a AST tree

        Args:
            string: The string we wish to convert into an AST
            root_type_name: The name of the type which is the first type me must
                choose on at the root of the tree.
            root_parser_name: An optional TypeParser name which overrides the
                default parser for the root_type_name.
            allow_partial_consume: Ordinarily we expect the parsing to fully consume
                the input string. This acts a sanity check for some parsers.
                However, for some tests we might be ok with not consuming everything.
        """
        #print(f"stringparser PARSE {string}")
        root_parser = _get_root_parser(self._type_context, root_type_name, root_parser_name)
        if not root_parser:
            raise ValueError(f"Unable to get a parser type {root_type_name} and "
                             f"root_parser {root_parser_name}")
        root_type = self._type_context.get_type_by_name(root_type_name)
        new_node, string_metadata = self._parse_object_choice_node(
            string, root_parser, root_type)
        if string_metadata.remaining_right_starti != len(string) and not allow_partial_consume:
            raise AInixParseError(
                f"Error. Expect to fully consume input string '{string}'. However, "
                f"only consumed {string[:string_metadata.remaining_right_starti]}")
        return new_node

    def _parse_object_node(
        self,
        implementation: typecontext.AInixObject,
        string: str,
        parser: ainix_common.parsing.parse_primitives.ObjectParser,
        origional_offset: int
    ) -> Tuple[ObjectNode, ParseDelegationReturnMetadata]:
        """Parses a string into a ObjectNode"""
        object_parse, delegation_to_node_map = self._run_object_parser_with_delegations(
            string, implementation, parser)
        arg_name_to_node: Dict[str, ObjectChoiceNode] = {}
        for arg in implementation.children:
            arg_present_data = object_parse.get_arg_present(arg.name)
            arg_name_to_node[arg.name], arg_metadata = \
                self._make_node_for_arg(arg, arg_present_data, delegation_to_node_map)
        my_return_metadata = ParseDelegationReturnMetadata(
            True, string, origional_offset, implementation, object_parse.remaining_start_i)
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
                out_delegation_return = ParseDelegationReturnMetadata(
                    parse_metadata.parse_success, parse_metadata.string_parsed,
                    delegation.slice_to_parse[0], arg, parse_metadata.remaining_right_starti,
                    parse_metadata.fail_reason
                )
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
                delegation.string_to_parse, arg, delegation.slice_to_parse[0])

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
        arg_has_already_been_delegated = arg_is_present and \
            arg_data.set_from_delegation is not None
        if arg_has_already_been_delegated:
            done_delegation = arg_data.set_from_delegation
            return delegation_to_node_map[done_delegation], done_delegation

        if not arg.required:
            arg_string_metadata = None
            if arg_is_present:
                arg_has_substructure_to_parse = arg.type_name is not None
                if arg_has_substructure_to_parse:
                    # TODO (DNGros): add back in stripping for slice_string?
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
        if parser is None:
            raise ValueError(f"Attempt to parse object {implementation.name} but"
                             f"no parser given.")
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
        ei = max(si, child_metadata.original_start_offset + child_metadata.remaining_right_starti)
        #if child_metadata.original_start_offset != 0:
        #    raise ValueError("Is this ever non 0???")
        #ei = max(endi, si + child_metadata.remaining_right_starti)
        return ParseDelegationReturnMetadata(True, result.string, 0, result.type, ei)

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
                result.get_implementation(),
                result.get_next_string(),
                result.next_parser,
                result.get_next_slice()[0]
            )
        metadata = self._object_choice_result_to_string_metadata(result, child_string_metadata)
        return ObjectChoiceNode(type, next_object_node), metadata

    def _run_type_parser_with_delegations(
        self,
        string: str,
        type_to_parser: typecontext.AInixType,
        parser: ainix_common.parsing.parse_primitives.TypeParser
    ) -> Tuple[
        TypeParserResult,
        Dict[ParseDelegationReturnMetadata, Tuple[ObjectNode, ParseDelegationReturnMetadata]]
    ]:
        """Run a type parser handling delegations it yields as needed."""
        parser_gen = parser.parse_string(string, type_to_parser)
        delegation_to_node: Dict[ParseDelegationReturnMetadata, ObjectNode] = {}
        last_delegation_result = None
        while True:
            try:
                delegation = parser_gen.send(last_delegation_result)
            except StopIteration as stop_iter:
                return_result = stop_iter.value
                return return_result, delegation_to_node
            # Well it didn't reach the end, so it must have yielded a delegation. Handle that.
            node, return_metadata = self._delegate_impl_parser(delegation)
            delegation_to_node[return_metadata] = node, return_metadata
            last_delegation_result = return_metadata

    def _delegate_impl_parser(self, delegation: ImplementationParseDelegation):
        try:
            node, return_metadata = self._parse_object_node(
                delegation.implementation,
                delegation.string_to_parse,
                delegation.next_parser,
                delegation.slice_to_parse[0]
            )
            return node, return_metadata
        except AInixParseError as e:
            fail_return = ParseDelegationReturnMetadata(
                False, delegation.string_to_parse, None, delegation.implementation, None, str(e))
            return None, fail_return


class AstUnparser:
    """A class which is the main driver for converting from ASTs to strings. It
    uses the to_string methods on TypeParsers and ObjectParsers

    Args:
        type_context: The type context to unparse in
        default_input_str_tokenizer: The tokenizer to use when trying to unparse
            something with copytokens. This should match the same tokenizer that
            was used when creating the input ast in order for things to match
            correctly.
    """
    def __init__(
        self,
        type_context: typecontext.TypeContext,
        default_input_str_tokenizer: tokenizers.StringTokenizer = None
    ):
        self._type_context = type_context
        self.input_str_tokenizer = default_input_str_tokenizer

    def _unparse_object_choice_node(
        self,
        node: ObjectChoiceNode,
        parser: TypeParser,
        result_builder: '_UnparseResultBuilder',
        left_offset: int,
        child_nums_here: Tuple[int, ...]
    ) -> str:
        if node.copy_was_chosen:
            out_string = result_builder.add_from_copy_node(
                node.next_node_is_copy, child_nums_here + (0, ), left_offset)
            result_builder.add_subspan(node, child_nums_here, out_string, left_offset)
            return out_string
        unparse = parser.to_string(node.next_node_not_copy.implementation, node.type_to_choose)
        out_string = ""
        new_left_offset = left_offset
        for part_of_out in unparse.unparse_seq:
            if isinstance(part_of_out, str):
                out_string += part_of_out
                new_left_offset += len(part_of_out)
            elif isinstance(part_of_out, ImplementationToStringDelegation):
                impl_string = self._unparse_object_node(
                    node.next_node_not_copy, part_of_out.next_parser, result_builder,
                    new_left_offset, child_nums_here + (0, ))
                out_string += impl_string
                new_left_offset += len(impl_string)
            else:
                raise ValueError("Unexpected object in unparse_seq")
        result_builder.add_subspan(node, child_nums_here, out_string, left_offset)
        return out_string

    def _unparse_optional_obj_choice_node(
        self,
        node: ObjectChoiceNode,
        parser_actual_type: TypeParser,
        result_builder: '_UnparseResultBuilder',
        left_offset: int,
        child_nums_here: Tuple[int, ...]
    ) -> str:
        """Unparses an ObjectChoiceNode for an arg present or not. Does not actually
        use a parser and passes through the value to the next parser"""
        if node.copy_was_chosen:
            out_string = result_builder.add_from_copy_node(
                node.next_node_is_copy, child_nums_here + (0, ), left_offset)
            result_builder.add_subspan(node, child_nums_here, out_string, left_offset)
            return out_string
        elif is_obj_choice_a_present_node(node):
            next_node = node.next_node_not_copy.get_choice_node_for_arg(
                OPTIONAL_ARGUMENT_NEXT_ARG_NAME)
            arg_str = self._unparse_object_choice_node(
                next_node, parser_actual_type, result_builder, left_offset,
                child_nums_here + (0, 0))
            result_builder.add_subspan(node.next_node, child_nums_here + (0, ),
                                       arg_str, left_offset)
        elif is_obj_choice_a_not_present_node(node):
            arg_str = ""
        else:
            raise ValueError("Expected a present node choice here")
        result_builder.add_subspan(node, child_nums_here, arg_str, left_offset)
        return arg_str

    def _unparse_object_node(
        self,
        node: ObjectNode,
        parser: ObjectParser,
        result_builder: '_unparseresultbuilder',
        left_offset: int,
        child_nums_here: Tuple[int, ...]
    ) -> str:
        out_string = ""
        new_left_offset = left_offset
        unparse_result = parser.to_string(self._obj_node_to_arg_map(node))
        for part_of_out in unparse_result.unparse_seq:
            if isinstance(part_of_out, str):
                out_string += part_of_out
                new_left_offset += len(part_of_out)
            elif isinstance(part_of_out, ArgIsPresentToString):
                out_string += part_of_out.string
                new_left_offset += len(part_of_out.string)
            elif isinstance(part_of_out, ArgToStringDelegation):
                next_node = node.get_choice_node_for_arg(part_of_out.arg.name)
                child_ind = node.implementation.children.index(part_of_out.arg)
                new_child_path = child_nums_here + (child_ind, )
                if part_of_out.arg.required:
                    arg_string = self._unparse_object_choice_node(
                        next_node, part_of_out.arg.type_parser, result_builder, new_left_offset,
                        new_child_path)
                else:
                    arg_string = self._unparse_optional_obj_choice_node(
                        next_node, part_of_out.arg.type_parser, result_builder, new_left_offset,
                        new_child_path)
                out_string += arg_string
                new_left_offset += len(arg_string)
            else:
                raise ValueError(f"Unexpected object in unparse_seq {part_of_out} of type "
                                 f"{part_of_out.__class__}")
        # TODO (DNGros): Loop through all not present args and unparse them to add their span.
        # This will allow us to add back in checks into copytools to make sure everything unparses
        result_builder.add_subspan(node, child_nums_here, out_string, left_offset)
        return out_string

    def _obj_node_to_arg_map(self, node: ObjectNode) -> ObjectNodeArgMap:
        is_present_map = {}
        for arg in node.implementation.children:
            chosen = node.get_choice_node_for_arg(arg.name)
            is_present_map[arg.name] = chosen.copy_was_chosen or \
                                       arg.required or \
                                       chosen.get_chosen_impl_name() != arg.not_present_object.name

        return ObjectNodeArgMap(
            implenetation=node.implementation,
            is_present_map=is_present_map
        )

    @functools.lru_cache(maxsize=500)
    def to_string(
        self,
        ast: ObjectChoiceNode,
        copy_from_str: str = None,
        root_parser_name: str = None
    ) -> 'UnparseResult':
        """The main method used to convert an AST into a string

        Args:
            ast: The ast to convert into a string form
            copy_from_str: The string to refer to when unparsing copy nodes
            root_parser_name: Overrides the unparsing to use a certain type
                parser for the root of the ast.
        """
        if copy_from_str:
            _, tokenized_metadata = self.input_str_tokenizer.tokenize(copy_from_str)
        else:
            tokenized_metadata = None

        root_parser = _get_root_parser(
            self._type_context,
            ast.get_type_to_choose_name(),
            root_parser_name
        )
        if not root_parser:
            raise ValueError(f"Unable to get a parser type {ast.get_type_to_choose_name()} and "
                             f"root_parser {root_parser_name}")
        result_builder = _UnparseResultBuilder(ast, tokenized_metadata)
        self._unparse_object_choice_node(ast, root_parser, result_builder, 0, tuple())
        return result_builder.as_result()


@attr.s(auto_attribs=True, frozen=True)
class UnparseResult:
    """Used to keep track about the relations of AST nodes to their unparsed
    string representation."""
    total_string: str
    child_path_and_node_to_span: Mapping[
        Tuple[Tuple[int, ...], Union[ObjectNode, ObjectChoiceNode]],
        Tuple[int, int]
    ]

    def pointer_to_span(self, pointer: AstIterPointer) -> Tuple[int, int]:
        span = self.child_path_and_node_to_span.get(
            (pointer.get_child_nums_here(), pointer.cur_node))
        return span

    def pointer_to_string(self, pointer: AstIterPointer) -> str:
        span = self.pointer_to_span(pointer)
        if span is None:
            return None
        si, endi = span
        return self.total_string[si:endi]


class _UnparseResultBuilder:
    """Used in AstUnparser while building a result"""
    def __init__(self, root: ObjectChoiceNode, copy_input_token_data: StringTokensMetadata):
        self.root = root
        self._node_map: Dict[
            Tuple[Tuple[int, ...], Union[ObjectNodeLike, ObjectChoiceNode]],
            Tuple[str, int]
        ] = {}
        self._copy_input_token_data = copy_input_token_data

    def add_subspan(
        self,
        node: Union[ObjectNodeLike, ObjectChoiceNode],
        child_nums_path: Tuple[int, ...],
        string: str,
        left_offset: int
    ):
        """Adds adds a unparsed string

        Args:
            node: the node that this string relates to
            child_nums_path: the child num paths that comes from pointer.get_child_nums_here
            string: the string which we have unparsed as
            left_offset: the offset into the unparsed string this occurs
        """
        self._node_map[child_nums_path, node] = (string, left_offset)

    def as_result(self) -> UnparseResult:
        top_string, _ = self._node_map[tuple(), self.root]
        node_to_span = pmap({
            node_info: (left_offset, left_offset+len(string))
            for node_info, (string, left_offset) in self._node_map.items()
        })
        return UnparseResult(top_string, node_to_span)

    def add_from_copy_node(
        self,
        node: CopyNode,
        child_path_here: Tuple[int, ...],
        left_offset: int
    ) -> str:
        """Used to add a span from a CopyNode appropriately copy from the input string

        Returns:
            The string that was copied over
        """
        if self._copy_input_token_data is None:
            raise RuntimeError("No copy string data given for tree with copy nodes")
        string = "".join(self._copy_input_token_data.joinable_tokens[
            self._copy_input_token_data.actual_pos_to_joinable_pos[node.start]:
            self._copy_input_token_data.actual_pos_to_joinable_pos[node.end] + 1
        ])
        self.add_subspan(node, child_path_here, string, left_offset)
        return string


def _get_root_parser(
    type_context: typecontext.TypeContext,
    type_name: str,
    parser_name: Optional[str]
) -> ainix_common.parsing.parse_primitives.TypeParser:
    """The public interface accepts a type name to parse as the root AST node
    for the string. This method converts that type name into an actuall parser
    instance."""
    if parser_name:
        return type_context.get_type_parser_by_name(parser_name)
    else:
        type_instance = type_context.get_type_by_name(type_name)
        if type_instance.default_type_parser is None:
            raise ValueError(f"No default type parser available for {type_instance}")
        return type_instance.default_type_parser
