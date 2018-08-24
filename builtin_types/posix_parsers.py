import parse_primitives
from typing import List


def lex_bash(string: str) -> List[tuple]:
    inside_single_quotes = False
    inside_double_quotes = False
    last_was_slash_escape = False
    current_word = []
    current_word_start = 0
    output = []
    for i, c in enumerate(string):
        if c == "'" and not inside_double_quotes and not last_was_slash_escape:
            inside_single_quotes = not inside_double_quotes
        if c == '"' and not inside_single_quotes and not last_was_slash_escape:
            inside_double_quotes = not inside_double_quotes
        last_was_slash_escape = c == "\\" and not last_was_slash_escape
        if c == " " and not inside_single_quotes and not inside_double_quotes:
            if len(current_word) > 0:
                output.append(("".join(current_word), (current_word_start, i+1)))
            current_word_start = i+1
            current_word = []
        else:
            current_word.append(c)
    if len(current_word) > 0:
        output.append(("".join(current_word), (current_word_start, len(string))))
    return output


####
def CmdSeqParser(
    run: parse_primitives.ObjectParserRun,
    string: str,
    result: parse_primitives.ObjectParserResult
):
    if string == "":
        raise parse_primitives.AInixParseError("Unable to parse empty string")
    operator_index = _get_location_of_operator(string)
    if operator_index is None:
        result.set_arg_present("ProgramArg", 0, len(string))
    else:
        result.set_arg_present("ProgramArg", 0, operator_index)
        result.set_arg_present("CompoundOp", operator_index, len(string))


def _get_location_of_operator(string: str):
    inside_single_quotes = False
    inside_double_quotes = False
    last_was_slash_escape = False
    for i, c in enumerate(string):
        if c == "|" and not inside_double_quotes and not inside_single_quotes:
            return i
        if c == "'" and not inside_double_quotes and not last_was_slash_escape:
            inside_single_quotes = not inside_double_quotes
        if c == '"' and not inside_single_quotes and not last_was_slash_escape:
            inside_double_quotes = not inside_double_quotes
        last_was_slash_escape = c == "\\" and not last_was_slash_escape
    return None
####


def ProgramTypeParser(
    run: parse_primitives.TypeParserRun,
    string: str,
    result: parse_primitives.TypeParserResult
):
    first_word = string.split(" ")[0]
    matching_programs = run.match_attribute(
        run.all_type_implementations, "invoke_name", first_word)
    if matching_programs:
        result.set_valid_implementation(matching_programs[0])
        first_space_index = string.find(" ")
        if first_space_index == -1:
            first_space_index = len(string)
        result.set_next_slice(first_space_index, len(string))
    else:
        raise parse_primitives.AInixParseError("Unable to find program", first_word)
####


def get_arg_with_short_name(arg_list, short_name: str):
    assert len(short_name) == 1, "unexpectedly long short_name"
    matches = [a for a in arg_list if a.arg_data['short_name'] == short_name]
    if not matches:
        return None
    if len(matches) > 1:
        raise parse_primitives.AInixParseError(
            "Program has multiple args with same short_name", matches)
    return matches[0]


def _get_arg_with_long_name(
    arg_list,
    long_name: str
):
    exact_matches = [arg for arg in arg_list
                     if arg.arg_data.get('long_name', None) == long_name]
    if exact_matches:
        if len(exact_matches) > 1:
            raise parse_primitives.AInixParseError("Program has multiple args "
                                                   "with same long_name",
                                                   exact_matches)
        return exact_matches[0]
    else:
        # TODO (DNGros): implment "non-ambigious abbriviations" for args
        return None


def _get_next_slice_of_lex_result(lex_result, current_index):
    # If nothing remaining, use next word
    no_next_word = current_index == len(lex_result) - 1
    if no_next_word:
        raise parse_primitives.AInixParseError(
            "Unable to find value for arg that requires one")
    next_lex_slice = lex_result[current_index + 1][1]
    return next_lex_slice


def ProgramObjectParser(
    run: parse_primitives.ObjectParserRun,
    string: str,
    result: parse_primitives.ObjectParserResult
):
    remaining_args = list(run.all_arguments)
    parameter_end_seen = False
    lex_result = lex_bash(string)
    for (lex_index, (word, (start_idx, end_idx))) in enumerate(lex_result):
        if word == "--":
            parameter_end_seen = True
        single_dash = not parameter_end_seen and len(word) >= 2 and \
                      word[0] == "-" and word[1] != "-"
        double_dash = not parameter_end_seen and len(word) >= 3 and word[:2] == "--"
        # TODO (DNGros): handle args that consume multiple words
        if single_dash:
            for ci, char in enumerate(word[1:]):
                shortname_match = get_arg_with_short_name(remaining_args, char)
                if shortname_match:
                    requires_value = shortname_match.type is not None
                    use_start_idx, use_end_idx = 0, 0
                    if requires_value:
                        remaining_chars = ci < len(word[1:]) - 1
                        if remaining_chars:
                            # Assume rest of chars are the value
                            use_start_idx = start_idx + (ci+1) + 1  # +1 for the dash
                            use_end_idx = end_idx
                        else:
                            next_slice = _get_next_slice_of_lex_result(
                                lex_result, lex_index)
                            use_start_idx, use_end_idx = next_slice
                    result.set_arg_present(shortname_match.name,
                                           use_start_idx, use_end_idx)
                    remaining_args.remove(shortname_match)
        elif double_dash:
            long_name_match = _get_arg_with_long_name(
                remaining_args, word[2:])
            if long_name_match:
                requires_value = long_name_match.type is not None
                use_start_idx, use_end_idx = 0, 0
                if requires_value:
                    next_slice = _get_next_slice_of_lex_result(
                        lex_result, lex_index)
                    use_start_idx, use_end_idx = next_slice
                result.set_arg_present(long_name_match.name,
                                       use_start_idx, use_end_idx)
                remaining_args.remove(long_name_match)

    # TODO (DNGros): handle positional args
    remaining_required_args = [a for a in remaining_args if a.required]
    if remaining_required_args:
        raise parse_primitives.AInixParseError(
            "Unexpected unmatched args", remaining_required_args)
####


def CommandOperatorParser(parser, string, result):
    pass

####


def CommandOperatorObjParser(parser, string, result):
    pass
