import pytest
from parsing.posix_parseing import *
from typegraph import TypeGraph


@pytest.fixture(scope="function")
def type_graph():
    tg = TypeGraph()
    init(tg)
    return tg


def test_cmd_seq_parser(type_graph):
    instance = CmdSeqParser(type_graph.get_type_by_name("CommandSequenceO"))
    result = instance.parse_string("hello")
    assert result.get_arg_present("program").slice == (0, 5)
    assert result.get_sibling_arg() is None

    result = instance.parse_string("woo | foo")
    assert result.get_arg_present("program").slice == (0, 4)
    assert result.get_arg_present("program").slice_string == "woo"
    assert result.get_sibling_arg().slice == (4, 9)
    assert result.get_sibling_arg().slice_string == "| foo"

    result = instance.parse_string("woo | foo | go")
    assert result.get_arg_present("program").slice == (0, 4)
    assert result.get_arg_present("program").slice_string == "woo"
    assert result.get_sibling_arg().slice == (4, 14)
    assert result.get_sibling_arg().slice_string == "| foo | go"


def test_cmd_seq_parser_quotes(type_graph):
    instance = CmdSeqParser(type_graph.get_type_by_name("CommandSequenceO"))
    result = instance.parse_string('hello "pi |" | foo')
    assert result.get_arg_present("program").slice == (0, 13)
    assert result.get_arg_present("program").slice_string == 'hello "pi |"'
    assert result.get_sibling_arg().slice == (13, 18)
    assert result.get_sibling_arg().slice_string == "| foo"

    result = instance.parse_string(r'he "\"pi\" |" | foo')
    assert result.get_arg_present("program").slice == (0, 14)
    assert result.get_arg_present("program").slice_string == r'he "\"pi\" |"'
    assert result.get_sibling_arg().slice == (14, 19)
    assert result.get_sibling_arg().slice_string == "| foo"


def test_prog_type_parser(type_graph):
    instance = ProgramTypeParser(type_graph.get_type_by_name("Program"))
    type_graph.create_object("FooProgram", "Program", [], type_data={"invoke_name": "foo"})
    type_graph.create_object("BarProgram", "Program", [], type_data={"invoke_name": "bar"})

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


def test_prog_object_parser_nocrash(type_graph):
    noargs = type_graph.create_object("FooProgram", "Program", [])
    instance = ProgramObjectParser(noargs)
    instance.parse_string("")


def test_short_name_match(type_graph):
    result = ProgramObjectParser._get_arg_with_short_name(
        [AInixArgument("a", None, arg_data={"short_name" : "a"})], "a")
    assert result.name == "a"

    result = ProgramObjectParser._get_arg_with_short_name(
        [AInixArgument("b", None, arg_data={"short_name" : "b"}),
         AInixArgument("a", None, arg_data={"short_name" : "a"})], "a")
    assert result.name == "a"


def test_prog_object_parser(type_graph):
    onearg = type_graph.create_object(
        "FooProgram", "Program",
        [AInixArgument("a", None, arg_data = {"short_name": "a"})])
    instance = ProgramObjectParser(onearg)
    result = instance.parse_string("-a")
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == ""


def test_prog_object_parser2(type_graph):
    type_graph.create_type("FooType")
    twoarg = type_graph.create_object(
        "FooProgram", "Program",
        [AInixArgument("a", "FooType", arg_data = {"short_name": "a"})])
    instance = ProgramObjectParser(twoarg)
    result = instance.parse_string("-a hello")
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == "hello"


def test_prog_object_parser3(type_graph):
    type_graph.create_type("FooType")
    twoarg = type_graph.create_object(
        "FooProgram", "Program",
        [AInixArgument("a", "FooType", arg_data = {"short_name": "a"})])
    instance = ProgramObjectParser(twoarg)
    result = instance.parse_string("-ahela")
    assert result.get_arg_present("a") is not None
    assert result.get_arg_present("a").slice_string == "hela"

