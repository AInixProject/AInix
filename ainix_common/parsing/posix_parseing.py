from parse_primitives import AInixType, AInixObject, AInixArgument, \
    ObjectParser, AInixParseError, ValueParser

CompoundOperator = AInixType("CompoundOperator")
CommandSequenceType = AInixType("CommandSequence")
ProgramType = AInixType("Program")

operators = [
    AInixObject(op_name, CompoundOperator,
                [AInixArgument("nextCommand", CommandSequenceType, required=True)])
    for op_name in ("pipe","and","or")
]

CommandSequenceObj = AInixObject(
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

