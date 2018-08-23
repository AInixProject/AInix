import typecontext
import parse_primitives
import re


def max_munch_type_parser(
    parser: parse_primitives.TypeParser,
    string: str,
    result: parse_primitives.TypeParserResult
) -> None:
    implementations = parser.type_implementations
    longest_match = (0, None)
    for implementation in implementations:
        parse_rep: str = implementation.type_data['ParseRepresentation']
        if string.startswith(parse_rep):
            match = (len(parse_rep), implementation)
            if match > longest_match:
                longest_match = match
    if longest_match[1] is None:
        raise parse_primitives.AInixParseError("Unable to find any matches "
                                               "in a max munch parser")
    result.set_valid_implementation(longest_match[1])
    result.set_next_slice(0, longest_match[0])


def regex_group_object_parser(
    parser: parse_primitives.ObjectParser,
    object: typecontext.AInixObject,
    string: str,
    result: parse_primitives.ObjectParserResult
) -> None:
    for arg in object.children:
        regex: str = arg.arg_data['RegexRepresentation']
        match = re.match(regex, string)
        arg_present = match is not None
        if arg_present:
            start_idx, end_idx = re.span(1)
            result.set_arg_present(arg.name, start_idx, end_idx)
        elif arg.required:
            raise parse_primitives.AInixParseError(
                f"Arg {arg.name} with RegexRepresentation {regex} did not "
                f"match on {string}, but the arg is required.")
