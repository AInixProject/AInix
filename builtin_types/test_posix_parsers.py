import pytest

from ainix_common.parsing.grammar_lang import create_object_parser_from_grammar
from builtin_types.posix_parsers import *
from unittest.mock import MagicMock

from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import AInixArgument, TypeContext, AInixObject, AInixType
from ainix_common.parsing.parse_primitives import AInixParseError, ObjectNodeArgMap, \
    ArgToStringDelegation, TypeParserResult
from ainix_common.parsing import loader


def _make_flag(short_name: str) -> AInixArgument:
    """Shorthand for constructing a flag argument"""
    return AInixArgument(MagicMock(), short_name, None, arg_data={SHORT_NAME: short_name},
                         parent_object_name=short_name + "fooparent")


def _make_positional(name="p1", position: int = 0,
                     multiword: bool = False, required: bool =False) -> AInixArgument:
    """Shorthand for constructing a positional argument"""
    return AInixArgument(MagicMock(), name, "FooType",
                         arg_data={POSITION: position, MULTIWORD_POS_ARG: multiword},
                         required=required)


@pytest.fixture(scope="function")
def type_context():
    context = TypeContext()
    import os
    dirname, filename = os.path.split(os.path.abspath(__file__))
    loader.load_path(f"{dirname}/command.ainix.yaml", context)
    return context


def gen_result(gen):
    """Get the result from a generator when expected to return next."""
    try:
        next(gen)
    except StopIteration as stp:
        return stp.value
    raise ValueError("Expected it to stop")


def test_cmd_seq_parser(type_context):
    cmd_seq_obj = type_context.get_object_by_name("CommandSequenceObj")
    instance = type_context.get_object_parser_by_name("CmdSeqParser")
    result = gen_result(instance.parse_string("hello", cmd_seq_obj))
    assert result.get_arg_present("ProgramArg").slice == (0, 5)
    assert result.get_arg_present("CompoundOp") is None

    result = gen_result(instance.parse_string("woo | foo", cmd_seq_obj))
    assert result.get_arg_present("ProgramArg").slice == (0, 3)
    assert result.get_arg_present("ProgramArg").slice_string == "woo"
    assert result.get_arg_present("CompoundOp").slice == (4, 9)
    assert result.get_arg_present("CompoundOp").slice_string == "| foo"

    result = gen_result(instance.parse_string("woo | foo | go", cmd_seq_obj))
    assert result.get_arg_present("ProgramArg").slice == (0, 3)
    assert result.get_arg_present("ProgramArg").slice_string == "woo"
    assert result.get_arg_present("CompoundOp").slice == (4, 14)
    assert result.get_arg_present("CompoundOp").slice_string == "| foo | go"


def test_cmd_seq_parser_quotes(type_context):
    cmd_seq_obj = type_context.get_object_by_name("CommandSequenceObj")
    instance = type_context.get_object_parser_by_name("CmdSeqParser")
    result = gen_result(instance.parse_string('hello "pi |" | foo', cmd_seq_obj))
    assert result.get_arg_present("ProgramArg").slice == (0, 12)
    assert result.get_arg_present("ProgramArg").slice_string == 'hello "pi |"'
    assert result.get_arg_present("CompoundOp").slice == (13, 18)
    assert result.get_arg_present("CompoundOp").slice_string == "| foo"

    result = gen_result(instance.parse_string(r'he "\"pi\" |" | foo', cmd_seq_obj))
    assert result.get_arg_present("ProgramArg").slice == (0, 13)
    assert result.get_arg_present("ProgramArg").slice_string == r'he "\"pi\" |"'
    assert result.get_arg_present("CompoundOp").slice == (14, 19)
    assert result.get_arg_present("CompoundOp").slice_string == "| foo"


def test_prog_type_parser(type_context):
    instance = type_context.get_type_parser_by_name("ProgramTypeParser")
    AInixObject(type_context, "FooProgram", "Program", [],
                type_data={"invoke_name": "foo"})
    AInixObject(type_context, "BarProgram", "Program", [],
                type_data={"invoke_name": "bar"})

    result = gen_result(instance.parse_string("foo -rm df"))
    assert result.get_implementation().name == "FooProgram"
    assert result.get_next_string() == "-rm df"
    result = gen_result(instance.parse_string("bar boop do"))
    assert result.get_implementation().name == "BarProgram"
    assert result.get_next_string() == "boop do"

    # Try a program which doesn't exist and get error
    with pytest.raises(AInixParseError):
        result = gen_result(instance.parse_string("baz bdsf do"))


def test_bash_lexer():
    result = lex_bash("foo bar")
    assert result == [("foo", (0, 4)), ("bar", (4, 7))]


def test_bash_lexer_quotes():
    result = lex_bash('foo "bar baz"')
    assert result == [("foo", (0, 4)), ('"bar baz"', (4, 13))]


def test_prog_object_parser_nocrash(type_context):
    noargs = AInixObject(type_context, "FooProgram", "Program", [])
    instance = type_context.get_object_parser_by_name("ProgramObjectParser")
    instance.parse_string("", noargs)


def test_short_name_match():
    mock_context = MagicMock()
    result = get_arg_with_short_name(
        [AInixArgument(mock_context, "a", None, arg_data={"short_name": "a"},
                       parent_object_name="a")], "a")
    assert result.name == "a"

    result = get_arg_with_short_name(
        [AInixArgument(mock_context, "b", None, arg_data={"short_name": "b"},
                       parent_object_name="safd"),
         AInixArgument(mock_context, "a", None, arg_data={"short_name": "a"},
                       parent_object_name="sdf")], "a")
    assert result.name == "a"


def test_prog_object_parser_basic(type_context):
    onearg = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data = {"short_name": "a"},
                       parent_object_name="sdf")])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("-a", onearg))
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == ""


def test_prog_object_tostring_basic(type_context):
    a_arg = AInixArgument(type_context, "a", None, arg_data={"short_name": "a"},
                          parent_object_name="sdf")
    onearg = AInixObject(
        type_context, "FooProgram", "Program",
        [a_arg])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    # Unparse
    unparse = parser.to_string(ObjectNodeArgMap(onearg, {"a": True}))
    assert unparse.unparse_seq == ["-a"]


def test_prog_object_tostring_basic_with_type(type_context):
    a_arg = AInixArgument(type_context, "a", "Program", arg_data={"short_name": "a"},
                          parent_object_name="sdf")
    onearg = AInixObject(
        type_context, "FooProgram", "Program",
        [a_arg])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    # Unparse
    unparse = parser.to_string(ObjectNodeArgMap(onearg, {"a": True}))
    assert unparse.unparse_seq == ["-a", " ", ArgToStringDelegation(a_arg)]


def test_prog_object_parser_argval(type_context):
    fooType = AInixType(type_context, "FooType")
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", fooType.name, arg_data={"short_name": "a"},
                       parent_object_name="sdf")])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("-a hello", argval))
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == "hello"
    assert result.remaining_start_i == len("-a hello")

    # Combined style
    result = gen_result(parser.parse_string("-afoo", argval))
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == "foo"


def test_prog_object_parser_twoargs(type_context):
    twoargs = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data={SHORT_NAME: "a"},
                       parent_object_name="sfd"),
         AInixArgument(type_context, "barg", None, arg_data={SHORT_NAME: "b"},
                       parent_object_name="sfd")]
    )
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("-a -b", twoargs))
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("barg") is not None
    result = gen_result(parser.parse_string("-b", twoargs))
    assert result.get_arg_present("a") is None
    assert result.get_arg_present("barg") is not None


def test_prog_object_parser_posarg(type_context):
    fooType = AInixType(type_context, "FooType")
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "p1", fooType.name, arg_data={POSITION: 0}, required=True)])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("hello", argval))
    assert result.get_arg_present("p1") is not None
    assert result.get_arg_present("p1").slice_string == "hello"


def test_prog_object_parser_2posarg(type_context):
    fooType = AInixType(type_context, "FooType")
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "p1", fooType.name, arg_data={POSITION: 0}, required=True),
         AInixArgument(type_context, "p2", fooType.name, arg_data={POSITION: 1}, required=True)])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("hello there", argval))
    assert result.get_arg_present("p1") is not None
    assert result.get_arg_present("p1").slice_string == "hello"
    assert result.get_arg_present("p2") is not None
    assert result.get_arg_present("p2").slice_string == "there"


def test_prog_object_parser_posarg_multword(type_context):
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [_make_positional(required=True, multiword=True)])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("hello yo there", argval))
    assert result.get_arg_present("p1") is not None
    assert result.get_arg_present("p1").slice_string == "hello yo there"
    assert result.remaining_start_i == len("hello yo there")


def test_prog_object_parser_posarg_multword_and_nonpos(type_context):
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [_make_positional(required=True, multiword=True),
         _make_flag("a")])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("hello yo -a", argval))
    assert result.get_arg_present("p1") is not None
    assert result.get_arg_present("p1").slice_string == "hello yo"
    assert result.get_arg_present("a") is not None
    assert result.remaining_start_i == len("hello yo -a")


def test_prog_object_parser_posarg_multword_and_nonpos_escape(type_context):
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [_make_positional(required=True, multiword=True),
         _make_flag("a")])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("-- hello yo -a", argval))
    assert result.get_arg_present("p1") is not None
    assert result.get_arg_present("p1").slice_string == "hello yo -a"
    assert result.get_arg_present("a") is None
    assert result.remaining_start_i == len("-- hello yo -a")


def test_prog_object_parser_posarg_multword_and_nonpos2(type_context):
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [_make_positional(required=True, multiword=True),
         _make_flag("a")])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("hello -a", argval))
    assert result.get_arg_present("p1") is not None
    assert result.get_arg_present("p1").slice_string == "hello"
    assert result.get_arg_present("a") is not None


def test_prog_object_parser_2posarg_multword(type_context):
    fooType = AInixType(type_context, "FooType")
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "p1", fooType.name,
                       arg_data={POSITION: 0, MULTIWORD_POS_ARG: True}, required=True),
         AInixArgument(type_context, "p2", fooType.name,
                       arg_data={POSITION: 1}, required=True)
         ])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("hello yo there woo", argval))
    assert result.get_arg_present("p1") is not None
    assert result.get_arg_present("p1").slice_string == "hello yo there"
    assert result.get_arg_present("p2") is not None
    assert result.get_arg_present("p2").slice_string == "woo"


def test_prog_object_parser_2posarg_multword_end(type_context):
    fooType = AInixType(type_context, "FooType")
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "p1", fooType.name,
                       arg_data={POSITION: 0}, required=True),
         AInixArgument(type_context, "p2", fooType.name,
                       arg_data={POSITION: 1, MULTIWORD_POS_ARG: True}, required=True)
         ])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("hello yo there woo", argval))
    assert result.get_arg_present("p1") is not None
    assert result.get_arg_present("p1").slice_string == "hello"
    assert result.get_arg_present("p2") is not None
    assert result.get_arg_present("p2").slice_string == "yo there woo"


def test_string_parse_e2e(type_context):
    twoargs = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data={"short_name": "a"},
                       parent_object_name="sdf"),
         AInixArgument(type_context, "barg", None, arg_data={"short_name": "b"},
                       parent_object_name="bw")],
        type_data={"invoke_name": "hello"}
    )
    parser = StringParser(type_context)
    ast = parser.create_parse_tree("hello -a", "CommandSequence")
    unparser = AstUnparser(type_context)
    to_string = unparser.to_string(ast)
    assert to_string.total_string == "hello -a"


def test_string_parse_e2e_multiword(type_context):
    fooType = AInixType(type_context, "FooType")
    fo = AInixObject(type_context, "fo", "FooType", [],
                     preferred_object_parser_name=create_object_parser_from_grammar(
                         type_context,
                         "fooname", '"foo"'
                     ).name)
    twoargs = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data={"short_name": "a"},
                       parent_object_name="sdf"),
         AInixArgument(type_context, "barg", None, arg_data={"short_name": "b"},
                       parent_object_name="bw"),
         AInixArgument(type_context, "p1", "FooType", arg_data={"position": 0},
                       parent_object_name="sdf")],
        type_data={"invoke_name": "hello"}
    )
    type_context.finalize_data()
    parser = StringParser(type_context)
    ast = parser.create_parse_tree("hello -a foo", "CommandSequence")
    unparser = AstUnparser(type_context)
    to_string = unparser.to_string(ast)
    assert to_string.total_string == "hello -a foo"


def test_string_parse_e2e_multiword2(type_context):
    fooType = AInixType(type_context, "FooType")
    fo = AInixObject(type_context, "fo", "FooType", [],
                     preferred_object_parser_name=create_object_parser_from_grammar(
                         type_context,
                         "fooname", '"foo bar"'
                     ).name)
    twoargs = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data={"short_name": "a"},
                       parent_object_name="sdf"),
         AInixArgument(type_context, "p1", "FooType", arg_data={POSITION: 0,
                                                                MULTIWORD_POS_ARG: True},
                       parent_object_name="sdf")],
        type_data={"invoke_name": "hello"}
    )
    type_context.finalize_data()
    parser = StringParser(type_context)
    ast = parser.create_parse_tree("hello foo bar -a", "Program")
    unparser = AstUnparser(type_context)
    to_string = unparser.to_string(ast)
    assert to_string.total_string == "hello -a foo bar"


def test_command_operator_parser(type_context):
    instance = type_context.get_type_parser_by_name("CommandOperatorParser")
    s = "| wc -l"
    result: TypeParserResult = gen_result(instance.parse_string(s))
    assert result.get_implementation().name == "PipeObj"
    assert result.get_next_slice() == (2, len(s))


def test_command_operator_parser_left_space(type_context):
    instance = type_context.get_type_parser_by_name("CommandOperatorParser")
    s = "  && wc -l"
    result: TypeParserResult = gen_result(instance.parse_string(s))
    assert result.get_implementation().name == "AndObj"
    assert result.get_next_slice() == (5, len(s))


def test_string_parse_e2e_sequence(type_context):
    twoargs = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data={"short_name": "a"},
                       parent_object_name="sdf"),
         AInixArgument(type_context, "barg", None, arg_data={"short_name": "b"},
                       parent_object_name="bw")],
        type_data={"invoke_name": "hello"}
    )
    parser = StringParser(type_context)
    unparser = AstUnparser(type_context)
    string = "hello -a | hello -b"
    ast = parser.create_parse_tree(string, "CommandSequence")
    to_string = unparser.to_string(ast)
    assert to_string.total_string == string

    no_space = "hello -a|hello -b"
    ast = parser.create_parse_tree(no_space, "CommandSequence")
    to_string = unparser.to_string(ast)
    assert to_string.total_string == string
