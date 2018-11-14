import pytest
from posix_parsers import *
from unittest.mock import MagicMock

from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import AInixArgument, TypeContext, AInixObject, AInixType
from ainix_common.parsing.parse_primitives import AInixParseError, ObjectNodeArgMap, \
    ArgToStringDelegation
from ainix_common.parsing import loader


@pytest.fixture(scope="function")
def type_context():
    context = TypeContext()
    loader.load_path("command.ainix.yaml", context)
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
    assert result.get_arg_present("ProgramArg").slice == (0, 4)
    assert result.get_arg_present("ProgramArg").slice_string == "woo"
    assert result.get_arg_present("CompoundOp").slice == (4, 9)
    assert result.get_arg_present("CompoundOp").slice_string == "| foo"

    result = gen_result(instance.parse_string("woo | foo | go", cmd_seq_obj))
    assert result.get_arg_present("ProgramArg").slice == (0, 4)
    assert result.get_arg_present("ProgramArg").slice_string == "woo"
    assert result.get_arg_present("CompoundOp").slice == (4, 14)
    assert result.get_arg_present("CompoundOp").slice_string == "| foo | go"


def test_cmd_seq_parser_quotes(type_context):
    cmd_seq_obj = type_context.get_object_by_name("CommandSequenceObj")
    instance = type_context.get_object_parser_by_name("CmdSeqParser")
    result = gen_result(instance.parse_string('hello "pi |" | foo', cmd_seq_obj))
    assert result.get_arg_present("ProgramArg").slice == (0, 13)
    assert result.get_arg_present("ProgramArg").slice_string == 'hello "pi |"'
    assert result.get_arg_present("CompoundOp").slice == (13, 18)
    assert result.get_arg_present("CompoundOp").slice_string == "| foo"

    result = gen_result(instance.parse_string(r'he "\"pi\" |" | foo', cmd_seq_obj))
    assert result.get_arg_present("ProgramArg").slice == (0, 14)
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
        [AInixArgument(mock_context, "a", None, arg_data={"short_name": "a"})], "a")
    assert result.name == "a"

    result = get_arg_with_short_name(
        [AInixArgument(mock_context, "b", None, arg_data={"short_name": "b"}),
         AInixArgument(mock_context, "a", None, arg_data={"short_name": "a"})], "a")
    assert result.name == "a"


def test_prog_object_parser_basic(type_context):
    onearg = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data = {"short_name": "a"})])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("-a", onearg))
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == ""


def test_prog_object_tostring_basic(type_context):
    a_arg = AInixArgument(type_context, "a", None, arg_data={"short_name": "a"})
    onearg = AInixObject(
        type_context, "FooProgram", "Program",
        [a_arg])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    # Unparse
    unparse = parser.to_string(ObjectNodeArgMap(onearg, {"a": True}))
    assert unparse.unparse_seq == ["-a"]


def test_prog_object_tostring_basic_with_type(type_context):
    a_arg = AInixArgument(type_context, "a", "Program", arg_data={"short_name": "a"})
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
        [AInixArgument(type_context, "a", fooType.name, arg_data={"short_name": "a"})])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("-a hello", argval))
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == "hello"

    # Combined style
    result = gen_result(parser.parse_string("-afoo", argval))
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == "foo"


def test_prog_object_parser_twoargs(type_context):
    twoargs = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data={"short_name": "a"}),
         AInixArgument(type_context, "barg", None, arg_data={"short_name": "b"})]
    )
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = gen_result(parser.parse_string("-a -b", twoargs))
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("barg") is not None
    result = gen_result(parser.parse_string("-b", twoargs))
    assert result.get_arg_present("a") is None
    assert result.get_arg_present("barg") is not None


def test_string_parse_e2e(type_context):
    twoargs = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data={"short_name": "a"}),
         AInixArgument(type_context, "barg", None, arg_data={"short_name": "b"})],
        type_data={"invoke_name": "hello"}
    )
    parser = StringParser(type_context)
    ast = parser.create_parse_tree("hello -a", "CommandSequence")
    unparser = AstUnparser(type_context)
    to_string = unparser.to_string(ast)
    assert to_string.total_string == "hello -a"

