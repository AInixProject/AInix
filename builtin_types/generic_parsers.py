from ainix_common.parsing import parse_primitives
import re
MAX_MUNCH_LOOKUP_KEY = "ParseRepresentation"
REGEX_GROUP_LOOKUP_KEY = "RegexRepresentation"


def max_munch_type_parser(
    run: parse_primitives.TypeParserRun,
    string: str,
    result: parse_primitives.TypeParserResult
) -> None:
    implementations = run.all_type_implementations
    longest_match = (0, None)
    for implementation in implementations:
        parse_rep: str = implementation.type_data[MAX_MUNCH_LOOKUP_KEY]
        if string.startswith(parse_rep):
            match = (len(parse_rep), implementation)
            if match > longest_match:
                longest_match = match
    if longest_match[1] is None:
        raise parse_primitives.AInixParseError(
            f"{run.parser_name} unable to find any matches inside {string}")
    result.set_valid_implementation(longest_match[1])
    result.set_next_slice(0, longest_match[0])


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
