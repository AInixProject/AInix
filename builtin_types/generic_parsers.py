from ainix_common.parsing import parse_primitives
import re

from ainix_common.parsing.parse_primitives import ParseDelegationReturnMetadata, UnparsableTypeError

REGEX_GROUP_LOOKUP_KEY = "RegexRepresentation"


def max_munch_type_parser(
    run: parse_primitives.TypeParserRun,
    string: str,
    result: parse_primitives.TypeParserResult
) -> None:
    """A type parser which consumes as much as possible."""
    implementations = run.all_type_implementations
    farthest_right = -9e9
    longest_success: ParseDelegationReturnMetadata = None
    for impl in implementations:
        deleg = yield run.delegate_parse_implementation(impl, (0, len(string)))
        if deleg.parse_success and deleg.remaining_right_starti > farthest_right:
            farthest_right = deleg.remaining_right_starti
            longest_success = deleg
    if not longest_success:
        raise UnparsableTypeError("A max munch parser did not find a valid implementation")
    result.accept_delegation(longest_success)


def max_munch_type_unparser(result: parse_primitives.TypeToStringResult):
    result.add_impl_unparse()


# Deprecated. Should eventually remove dependencies on
def regex_group_object_parser(
    run:  parse_primitives.ObjectParserRun,
    string: str,
    result: parse_primitives.ObjectParserResult
) -> None:
    for arg in run.all_arguments:
        regex: str = arg.arg_data[REGEX_GROUP_LOOKUP_KEY]
        match = re.match(regex, string)
        arg_present = match is not None
        if arg_present:
            start_idx, end_idx = match.span(1)
            result.set_arg_present(arg.name, start_idx, end_idx)
        elif arg.required:
            raise parse_primitives.AInixParseError(
                f'Arg {arg.name} with RegexRepresentation "{regex}" did not '
                f'match on "{string}", but the arg is required.')
