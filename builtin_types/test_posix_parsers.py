import pytest
from posix_parsers import *
from unittest.mock import MagicMock
from typecontext import AInixArgument, TypeContext, AInixObject, AInixType
from parse_primitives import AInixParseError
import loader


@pytest.fixture(scope="function")
def type_context():
    context = TypeContext()
    loader.load_path("command.ainix.yaml", context)
    return context


def test_cmd_seq_parser(type_context):
    cmd_seq_obj = type_context.get_object_by_name("CommandSequenceObj")
    instance = type_context.get_object_parser_by_name("CmdSeqParser")
    result = instance.parse_string(cmd_seq_obj, "hello")
    assert result.get_arg_present("program").slice == (0, 5)
    assert result.get_sibling_arg() is None

    result = instance.parse_string(cmd_seq_obj, "woo | foo")
    assert result.get_arg_present("program").slice == (0, 4)
    assert result.get_arg_present("program").slice_string == "woo"
    assert result.get_sibling_arg().slice == (4, 9)
    assert result.get_sibling_arg().slice_string == "| foo"

    result = instance.parse_string(cmd_seq_obj, "woo | foo | go")
    assert result.get_arg_present("program").slice == (0, 4)
    assert result.get_arg_present("program").slice_string == "woo"
    assert result.get_sibling_arg().slice == (4, 14)
    assert result.get_sibling_arg().slice_string == "| foo | go"


def test_cmd_seq_parser_quotes(type_context):
    cmd_seq_obj = type_context.get_object_by_name("CommandSequenceObj")
    instance = type_context.get_object_parser_by_name("CmdSeqParser")
    result = instance.parse_string(cmd_seq_obj, 'hello "pi |" | foo')
    assert result.get_arg_present("program").slice == (0, 13)
    assert result.get_arg_present("program").slice_string == 'hello "pi |"'
    assert result.get_sibling_arg().slice == (13, 18)
    assert result.get_sibling_arg().slice_string == "| foo"

    result = instance.parse_string(cmd_seq_obj, r'he "\"pi\" |" | foo')
    assert result.get_arg_present("program").slice == (0, 14)
    assert result.get_arg_present("program").slice_string == r'he "\"pi\" |"'
    assert result.get_sibling_arg().slice == (14, 19)
    assert result.get_sibling_arg().slice_string == "| foo"


def test_prog_type_parser(type_context):
    instance = type_context.get_type_parser_by_name("ProgramTypeParser")
    AInixObject(type_context, "FooProgram", "Program", [],
                type_data={"invoke_name": "foo"})
    AInixObject(type_context, "BarProgram", "Program", [],
                type_data={"invoke_name": "bar"})

    result = instance.parse_string("foo -rm df")
    assert result.get_implementation().name == "FooProgram"
    assert result.get_next_string() == "-rm df"
    result = instance.parse_string("bar boop do")
    assert result.get_implementation().name == "BarProgram"
    assert result.get_next_string() == "boop do"

    got_excep = False
    try:
        instance.parse_string("baz fdf wd")
    except AInixParseError:
        got_excep = True
    assert got_excep, "Expected to get a parse error"


def test_bash_lexer():
    result = lex_bash("foo bar")
    assert result == [("foo", (0,4)), ("bar", (4,7))]


def test_bash_lexer_quotes():
    result = lex_bash('foo "bar baz"')
    assert result == [("foo", (0,4)), ('"bar baz"', (4,13))]


def test_prog_object_parser_nocrash(type_context):
    noargs = AInixObject(type_context, "FooProgram", "Program", [])
    instance = type_context.get_object_parser_by_name("ProgramObjectParser")
    instance.parse_string(noargs, "")


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
    result = parser.parse_string(onearg, "-a")
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == ""


def test_prog_object_parser_argval(type_context):
    fooType = AInixType(type_context, "FooType")
    argval = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", fooType.name, arg_data={"short_name": "a"})])
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = parser.parse_string(argval, "-a hello")
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == "hello"

    # Combined style
    result = parser.parse_string(argval, "-afoo")
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == "foo"


def test_prog_object_parser_twoargs(type_context):
    twoargs = AInixObject(
        type_context, "FooProgram", "Program",
        [AInixArgument(type_context, "a", None, arg_data={"short_name": "a"}),
         AInixArgument(type_context, "barg", None, arg_data={"short_name": "b"})]
    )
    parser = type_context.get_object_parser_by_name("ProgramObjectParser")
    result = parser.parse_string(twoargs, "-a -b")
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("barg") is not None
    result = parser.parse_string(twoargs, "-b")
    assert result.get_arg_present("a") is None
    assert result.get_arg_present("barg") is not None
