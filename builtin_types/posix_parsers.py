"""
AInix parser implementations to process POSIX/GNU style commands.

https://www.gnu.org/software/libc/manual/html_node/Argument-Syntax.html

"""
from ainix_common.parsing import parse_primitives
from typing import List

from ainix_common.parsing.parse_primitives import AInixParseError
from ainix_common.parsing.typecontext import AInixArgument


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


def CmdSeqUnparser(
    arg_map: parse_primitives.ObjectNodeArgMap,
    result: parse_primitives.ObjectToStringResult
):
    if arg_map.is_argname_present("ProgramArg"):
        result.add_argname_tostring("ProgramArg")
    if arg_map.is_argname_present("CompoundOp"):
        result.add_string(" ")
        result.add_argname_tostring("CompoundOp")

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


def ProgramTypeUnparser(result: parse_primitives.TypeToStringResult):
    impl = result.implementation
    prog_name = impl.type_data['invoke_name']
    result.add_string(prog_name + " ")
    result.add_impl_unparse()
####


def get_arg_with_short_name(arg_list, short_name: str):
    assert len(short_name) == 1, "unexpectedly long short_name"
    matches = [a for a in arg_list if a.arg_data.get(SHORT_NAME, None) == short_name]
    if not matches:
        return None
    if len(matches) > 1:
        raise parse_primitives.AInixParseError(
            "Program has multiple args with same short_name", matches)
    return matches[0]


def get_all_positional_args(arg_list):
    """Returns all the positional args in order"""
    all_positional_args = [arg for arg in arg_list if POSITION in arg.arg_data]
    all_positional_args.sort(key=lambda arg: arg.arg_data[POSITION])
    return all_positional_args


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


SHORT_NAME = "short_name"
LONG_NAME = "long_name"
POSITION = "position"
MULTIWORD_POS_ARG = "multiword_pos_arg"

def ProgramObjectParser(
    run: parse_primitives.ObjectParserRun,
    string: str,
    result: parse_primitives.ObjectParserResult
):
    remaining_args = list(run.all_arguments)
    parameter_end_seen = False
    already_seen_multiword_positional = False
    lex_result = lex_bash(string)
    lex_index = 0
    while lex_index < len(lex_result):
        word, (start_idx, end_idx) = lex_result[lex_index]
        if word == "--":
            parameter_end_seen = True
        single_dash = not parameter_end_seen and len(word) >= 2 and \
                      word[0] == "-" and word[1] != "-"
        double_dash = not parameter_end_seen and len(word) >= 3 and word[:2] == "--"
        if single_dash:
            for ci, char in enumerate(word[1:]):
                shortname_match = get_arg_with_short_name(remaining_args, char)
                if shortname_match:
                    requires_value = shortname_match.type is not None
                    use_start_idx, use_end_idx = end_idx, end_idx
                    if requires_value:
                        # TODO (DNGros): handle args that consume multiple words
                        remaining_chars = ci < len(word[1:]) - 1
                        if remaining_chars:
                            # Assume rest of chars are the value
                            use_start_idx = start_idx + (ci+1) + 1  # +1 for the dash
                            use_end_idx = end_idx
                        else:
                            next_slice = _get_next_slice_of_lex_result(
                                lex_result, lex_index)
                            use_start_idx, use_end_idx = next_slice
                        lex_index += 1
                    result.set_arg_present(shortname_match.name,
                                           use_start_idx, use_end_idx)
                    remaining_args.remove(shortname_match)
            lex_index += 1
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
                    lex_index += 1
                result.set_arg_present(long_name_match.name,
                                       use_start_idx, use_end_idx)
                remaining_args.remove(long_name_match)
            lex_index += 1
        else:
            # Must be a positional arg
            sorted_pos_args = get_all_positional_args(remaining_args)
            if not sorted_pos_args:
                raise AInixParseError(f"Unexpected word '{word}' with no remaing positional args."
                                      f" Input was {string}")
            arg_to_do = sorted_pos_args[0]
            if arg_to_do.type is None:
                raise ValueError(f"Positional arg {arg_to_do.name} can't have None type")
            if SHORT_NAME in arg_to_do.arg_data or LONG_NAME in arg_to_do.arg_data:
                raise ValueError("Can't be both positional and a flag")
            is_multiword = arg_to_do.arg_data.get(MULTIWORD_POS_ARG, False) is True
            if is_multiword and already_seen_multiword_positional:
                raise ValueError("Cannot parse a multiword positional argument as a previous"
                                 "argument was also specified as multiword. Parse is ambigious")
            if is_multiword:
                already_seen_multiword_positional = True
                # We can potentially consume all the remaining words, except we have to leave
                # room any other positional args there might be.
                max_words_to_consume = len(lex_result) - lex_index - len(sorted_pos_args[1:])
                if not parameter_end_seen:
                    # If we haven't seen a "--" then we need to scan forwards to make sure we
                    # don't accidentally count a option flag as one of this positional arg.
                    words_to_consume = 1
                    while words_to_consume < max_words_to_consume:
                        lookahead_word, _ = lex_result[lex_index + words_to_consume]
                        if lookahead_word.startswith("-"):
                            # If we see an option arg, stop consuming words for this pos arg here
                            break
                        words_to_consume += 1
                else:
                    words_to_consume = max_words_to_consume
            else:
                words_to_consume = 1
            _, use_end_idx = lex_result[lex_index+words_to_consume-1][1]
            result.set_arg_present(arg_to_do.name, start_idx, use_end_idx)
            remaining_args.remove(arg_to_do)
            lex_index += words_to_consume

    remaining_required_args = [a for a in remaining_args if a.required]
    if remaining_required_args:
        raise parse_primitives.AInixParseError(
            "Unexpected unmatched args", remaining_required_args)


def _get_flag_for_arg_unparse(arg: AInixArgument) -> str:
    """Converts an arg into it's representive flag. Defaults to long style"""
    long_name = arg.arg_data.get(LONG_NAME, None)
    if long_name:
        return "--" + long_name
    short_name = arg.arg_data.get(SHORT_NAME, None)
    if not short_name:
        raise ValueError("No long name or short name on arg for a program")
    return "-" + short_name


def ProgramObjectUnparser(
    arg_map: parse_primitives.ObjectNodeArgMap,
    result: parse_primitives.ObjectToStringResult
):
    had_prev_args = False
    # First add in all the flag args
    for arg in arg_map.implementation.children:
        if not arg_map.is_argname_present(arg.name):
            continue
        if POSITION in arg.arg_data:
            continue
        if had_prev_args:
            result.add_string(" ")
        arg_flag = _get_flag_for_arg_unparse(arg)
        result.add_string(arg_flag)
        if arg.type is not None:
            result.add_string(" ")
            result.add_arg_tostring(arg)
        had_prev_args = True
    # Now do all the positional args in order
    for arg in get_all_positional_args(arg_map.implementation.children):
        if had_prev_args:
            result.add_string(" ")
        result.add_arg_tostring(arg)
        had_prev_args = True
####


OPERATOR_REP_KEY = "OperatorRepresentation"


def CommandOperatorParser(
    run: parse_primitives.TypeParserRun,
    string: str,
    result: parse_primitives.TypeParserResult
):
    lstriped = string.lstrip()
    amount_lstriped = len(string) - len(lstriped)
    canidates = [(o.type_data[OPERATOR_REP_KEY], o) for o in run.all_type_implementations
                 if lstriped.startswith(o.type_data[OPERATOR_REP_KEY])]
    if not canidates:
        raise AInixParseError(f"No operator matched string {string}")
    if len(canidates) > 2:
        raise ValueError(f"Operators shouldn't match more than twice? {string} {canidates}")
    # Sort taking longest first. So take || over |.
    canidates.sort(reverse=True)
    matching_str, matching_impl = canidates[0]
    result.set_valid_implementation(matching_impl)
    result.set_next_slice(len(matching_str) + amount_lstriped, len(string))


def CommandOperatorUnparserFunc(result: parse_primitives.TypeToStringResult):
    result.add_string(result.implementation.type_data[OPERATOR_REP_KEY])
    result.add_string(" ")
    result.add_impl_unparse()
