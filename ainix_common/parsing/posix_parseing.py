import shlex

from parse_primitives import ObjectParser, AInixParseError, \
    TypeParser, SingleTypeImplParser
from typegraph import AInixArgument


def init(typegraph):
    CommandSequenceType = typegraph.create_type("CommandSequence",
                                                default_type_parser=SingleTypeImplParser)
    CompoundOperator = typegraph.create_type("CompoundOperator",
                                             default_type_parser=CommandOperatorParser,
                                             default_object_parser=CommandOperatorObjParser)
    ProgramType = typegraph.create_type(
        "Program", default_type_parser=ProgramTypeParser,
        default_object_parser=ProgramObjectParser)

    operators = [
        typegraph.create_object(op_name, CompoundOperator,
                    [AInixArgument("nextCommand", CommandSequenceType, required=True)])
        for op_name in ("pipe","and","or") ]
    CommandSequenceObj = typegraph.create_object(
        "CommandSequenceO", CommandSequenceType,
        [AInixArgument("program", ProgramType, required=True)],
        AInixArgument("compoundOp", CompoundOperator))


def lex_bash(string: str) -> tuple:
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


class CmdSeqParser(ObjectParser):
    def _get_location_of_operator(self, string):
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
        
    def _parse_string(self, string, result):
        if string == "":
            raise AInixParseError("Unable to parse empty string")
        operator_index = self._get_location_of_operator(string)
        if operator_index is None:
            result.set_arg_present("program", 0, len(string))
        else:
            result.set_arg_present("program", 0, operator_index)
            result.set_sibling_present(operator_index, len(string))


class ProgramTypeParser(TypeParser):
    def _parse_string(self, string, result):
        first_word = string.split(" ")[0]
        matching_programs = self._match_attribute(
            self.type_implementations, "invoke_name", first_word)
        if matching_programs:
            result.set_valid_implementation(matching_programs[0])
            first_space_index = string.find(" ")
            if first_space_index == -1:
                first_space_index = len(string)
            result.set_next_slice(first_space_index, len(string))
        else:
            raise AInixParseError("Unable to find program", first_word)


class ProgramObjectParser(ObjectParser):
    @staticmethod
    def _get_arg_with_short_name(arg_list, short_name: str):
        assert len(short_name) == 1, "unexpectedly long short_name"
        matches = [a for a in arg_list if a.arg_data['short_name'] == short_name]
        if not matches:
            return None
        if len(matches) > 1:
            raise AInixParseError("Program has multiple args with same short_name", matches)
        return matches[0]

    @staticmethod
    def _get_arg_with_long_name(arg_list, long_name: str):
        exact_matches = arg_list.filter(
            lambda arg: arg.arg_data.get('long_name', None) == long_name)
        if exact_matches:
            if len(exact_matches) > 1:
                raise AInixParseError("Program has multiple args with same long_name",
                                      exact_matches)
            return exact_matches[0]
        else:
            # TODO (DNGros): implment "non-ambigious abbriviations" for args
            return None

    @staticmethod
    def _get_next_slice_of_lex_result(lex_result, current_index, arg):
        # If nothing remaining, use next word
        no_next_word = current_index == len(lex_result) - 1
        if no_next_word:
            raise AInixParseError(
                "Unable to find value for arg that requires one")
        next_lex_slice = lex_result[current_index + 1][1]
        return next_lex_slice

    def _parse_string(self, string, result):
        remaining_args = list(self.object.children)
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
                    shortname_match = self._get_arg_with_short_name(remaining_args, char)
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
                                next_slice = self._get_next_slice_of_lex_result(
                                    lex_result, lex_index, shortname_match)
                                use_start_idx, use_end_idx = next_slice
                        result.set_arg_present(shortname_match.name,
                                               use_start_idx, use_end_idx)
                        remaining_args.remove(shortname_match)
            elif double_dash:
                long_name_match = self._get_arg_with_long_name(
                    remaining_args, long_name_match)
                if long_name_match:
                    requires_value = shortname_match.type is not None
                    use_start_idx, use_end_idx = 0, 0
                    if requires_value:
                        next_slice = self._get_next_slice_of_lex_result(
                            lex_result, lex_index)
                        use_start_idx, use_end_idx = next_slice
                    result.set_arg_present(shortname_match.name,
                                           use_start_idx, use_end_idx, long_name_match)
                    remaining_args.remove(long_name_match)

        # TODO (DNGros): handle positional args
        remaining_required_args = [a for a in remaining_args if a.required]
        if remaining_required_args:
            raise AInixParseError("Unexpected unmatched args", remaining_required_args)


class CommandOperatorParser(TypeParser):
    pass


class CommandOperatorObjParser(ObjectParser):
    pass
