from parse_primitives import ObjectParser, AInixParseError, ValueParser
from typegraph import AInixArgument


def init(typegraph):
    CompoundOperator = typegraph.create_type("CompoundOperator")
    CommandSequenceType = typegraph.create_type("CommandSequence")
    ProgramType = typegraph.create_type("Program")

    operators = [
        typegraph.create_object(op_name, CompoundOperator,
                    [AInixArgument("nextCommand", CommandSequenceType, required=True)])
        for op_name in ("pipe","and","or")
    ]

    CommandSequenceObj = typegraph.create_object(
        "CommandSequenceO", CommandSequenceType,
        [AInixArgument("program", ProgramType, required=True)],
        AInixArgument("compoundOp", CompoundOperator))


class CmdSeqParser(ObjectParser):
    def _get_location_of_operator(self, string):
        inside_single_quotes = False
        inside_double_quotes = False
        lastWasSlashEscape = False
        lastChar = None
        for i, c in enumerate(string):
            if c == "|" and not inside_double_quotes and not inside_single_quotes:
                return i
            if c == "'" and not inside_double_quotes and not lastWasSlashEscape:
                inside_single_quotes = not inside_double_quotes
            if c == '"' and not inside_single_quotes and not lastWasSlashEscape:
                inside_double_quotes = not inside_double_quotes
            lastWasSlashEscape = c == "\\" and not lastWasSlashEscape
            lastChar = c
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


class ProgramTypeParser(ValueParser):
    def _parse_string(self, string, result):
        pass
