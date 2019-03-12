import pytest

from ainix_common.parsing import loader
from ainix_common.parsing.parse_primitives import AInixParseError
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.specialtypes import generic_strings

@pytest.fixture(scope="function")
def tc():
    context = TypeContext()
    loader.load_path("builtin_types/paths.ainix.yaml", context, up_search_limit=2)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml", context, up_search_limit=2)
    generic_strings.create_generic_strings(context)
    context.finalize_data()
    return context


@pytest.mark.parametrize("in_str", ["t", "txt", "fofoo", "tar.gz", "a.b.c"])
def test_path_parse_extension(tc, in_str):
    parser = StringParser(tc)
    ast = parser.create_parse_tree(in_str, "FileExtension")
    unparser = AstUnparser(tc)
    to_string = unparser.to_string(ast)
    assert to_string.total_string == in_str


@pytest.mark.parametrize("in_str", ("t",))
def test_dot_separated_words(tc, in_str):
    parser = StringParser(tc)
    ast = parser.create_parse_tree(in_str, "DotSeparatedWords")
    unparser = AstUnparser(tc)
    to_string = unparser.to_string(ast)
    assert to_string.total_string == in_str


@pytest.mark.parametrize("in_str",
    ("foo", "foo.txt", "foo/bar", "..", "../foo", "*", "~", "~/hello*/fdf.bar",
     ".dotfile", ".dotfile.txt", "1234.vhdl", "a.b.c", "a.b.c.d", ".shh/foo.txt",
     "foo.tar/bar.txt", "foo.txt/bar", "build/", "../../../", "./", "~foo",
     "foo.tar.gz"))
def test_path_parse_and_unparse_without_error(tc, in_str):
    parser = StringParser(tc)
    ast = parser.create_parse_tree(in_str, "Path")
    unparser = AstUnparser(tc)
    to_string = unparser.to_string(ast)
    assert to_string.total_string == in_str


@pytest.mark.parametrize("in_str",
                         ("~/foo/bar{1,2,3}",))
def test_path_parse_and_unparse_with_error(tc, in_str):
    parser = StringParser(tc)
    try:
        ast = parser.create_parse_tree(in_str, "Path")
        pytest.fail(f"{in_str} unexpectedly worked")
    except AInixParseError as e:
        pass


def test_empty_string_not_a_path(tc):
    parser = StringParser(tc)
    with pytest.raises(AInixParseError):
        ast = parser.create_parse_tree("", "Path")


@pytest.mark.parametrize("in_str",
                         ("foo.txt", "foo.txt bar.bat"))
def test_path_list_parse_and_unparse_without_error(tc, in_str):
    parser = StringParser(tc)
    ast = parser.create_parse_tree(in_str, "PathList")
    unparser = AstUnparser(tc)
    to_string = unparser.to_string(ast)
    assert to_string.total_string == in_str
