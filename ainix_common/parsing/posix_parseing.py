import parse_primitives
import typecontext


def init(typegraph) -> None:
    CommandSequenceType = typecontext.AInixType(typegraph, "CommandSequence",
                                                default_type_parser=parse_primitives.SingleTypeImplParser,
                                                default_object_parser=CmdSeqParser)
    CompoundOperator = typecontext.AInixType(typegraph, "CompoundOperator",
                                             default_type_parser=CommandOperatorParser,
                                             default_object_parser=CommandOperatorObjParser)
    ProgramType = typecontext.AInixType(
        typegraph, "Program", default_type_parser=ProgramTypeParser,
        default_object_parser=ProgramObjectParser)

    operators = [
        typecontext.AInixType(typegraph, op_name, CompoundOperator,
                              [typecontext.AInixArgument("nextCommand", CommandSequenceType, required=True)])
        for op_name in ("pipe","and","or")
    ]
    CommandSequenceObj = typegraph.create_object(
        "CommandSequenceO", CommandSequenceType,
        [typecontext.AInixArgument("program", ProgramType, required=True)],
        typecontext.AInixArgument("compoundOp", CompoundOperator))


