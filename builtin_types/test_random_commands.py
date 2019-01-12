import ainix_kernel
from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.model_specific.tokenizers import NonLetterTokenizer
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing import exampleloader
from ainix_kernel.specialtypes import allspecials
import pytest


@pytest.fixture()
def all_the_stuff_context():
    type_context = TypeContext()
    loader = TypeContextDataLoader(type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(type_context)

    with_example_files = ("numbers", "pwd", "ls", "cat", "head", "cp", "wc",
                          "mkdir", "echo", "mv", "touch")
    for f in with_example_files:
        loader.load_path(f"builtin_types/{f}.ainix.yaml")
    type_context.finalize_data()
    return type_context


def test_touch(all_the_stuff_context):
    tc = all_the_stuff_context
    parser = StringParser(tc)
    string = "touch foo.txt"
    ast = parser.create_parse_tree(string, "Program")
    unparser = AstUnparser(tc, NonLetterTokenizer())
    result = unparser.to_string(ast)
    assert result.total_string == string
    pointers = list(ast.depth_first_iter())
    assert result.pointer_to_string(pointers[0]) == string
    assert result.pointer_to_string(pointers[1]) == "foo.txt"


@pytest.mark.parametrize('string',
                         ("touch 1234.vhdl", 'touch -r store.json atlas_test.tar',
                          "touch ../.gitattributes"))
def test_touch2(all_the_stuff_context, string):
    tc = all_the_stuff_context
    parser = StringParser(tc)
    ast = parser.create_parse_tree(string, "Program")
    unparser = AstUnparser(tc, NonLetterTokenizer())
    result = unparser.to_string(ast)
    assert result.total_string == string
    pointers = list(ast.depth_first_iter())
    assert result.pointer_to_string(pointers[0]) == string
    #assert result.pointer_to_string(pointers[1]) == "foo.txt"

