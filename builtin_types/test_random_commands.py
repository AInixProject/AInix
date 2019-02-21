from pyrsistent import pmap

import ainix_kernel
from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode, ObjectNode, \
    CopyNode
from ainix_common.parsing.copy_tools import make_copy_version_of_tree, add_copies_to_ast_set
from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.model_specific.tokenizers import NonLetterTokenizer
from ainix_common.parsing.parse_primitives import AInixParseError
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing import exampleloader
from ainix_kernel.specialtypes import allspecials
import pytest

from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES


@pytest.fixture()
def all_the_stuff_context():
    type_context = TypeContext()
    loader = TypeContextDataLoader(type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(type_context)

    for f in ALL_EXAMPLE_NAMES:
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


@pytest.mark.parametrize('string',
                         ('cp foo bar', 'cp cug.tex ..'))
def test_cp(all_the_stuff_context, string):
    tc = all_the_stuff_context
    parser = StringParser(tc)
    ast = parser.create_parse_tree(string, "Program")
    unparser = AstUnparser(tc, NonLetterTokenizer())
    result = unparser.to_string(ast)
    assert result.total_string == string
    pointers = list(ast.depth_first_iter())
    assert result.pointer_to_string(pointers[0]) == string


def test_touch_set(all_the_stuff_context):
    tc = all_the_stuff_context
    parser = StringParser(tc)
    string = "touch script_window.hpp"
    ast = parser.create_parse_tree(string, "Program")
    unparser = AstUnparser(tc, NonLetterTokenizer())
    result = unparser.to_string(ast)
    assert result.total_string == string

    cset = AstObjectChoiceSet(tc.get_type_by_name("Program"))
    cset.add(ast, True, 1, 1)
    new_ast = parser.create_parse_tree(string, "Program")
    assert cset.is_node_known_valid(new_ast)

    tokenizer = NonLetterTokenizer()
    x_str = 'set the last mod time of script_window.hpp to now'
    _, tok_metadata = tokenizer.tokenize(x_str)
    ast_copies = make_copy_version_of_tree(ast, unparser, tok_metadata)
    add_copies_to_ast_set(ast, cset, unparser, tok_metadata)
    assert cset.is_node_known_valid(ast_copies)
    assert cset.is_node_known_valid(ast)

    # Scary complicated reconstruction of something that broke it
    # could be made into a simpler unit test in copy_tools
    touch_o = tc.get_object_by_name("touch")
    file_list = tc.get_type_by_name("PathList")
    r_arg = touch_o.get_arg_by_name("r")
    other_copy = ObjectChoiceNode(
        tc.get_type_by_name("Program"),
        ObjectNode(
            touch_o,
            pmap({
                "r": ObjectChoiceNode(r_arg.present_choice_type,
                                      ObjectNode(r_arg.not_present_object, pmap())),
                "file_list": ObjectChoiceNode(file_list, CopyNode(file_list, 12, 14))
            })
        )
    )
    other_result = unparser.to_string(
        other_copy, x_str)
    assert other_result.total_string == string
    assert cset.is_node_known_valid(other_copy)


@pytest.mark.parametrize('string',
                         ('asdf', "echo 'hi' | od -c"))
def test_fails(all_the_stuff_context, string):
    tc = all_the_stuff_context
    parser = StringParser(tc)
    with pytest.raises(AInixParseError):
        ast = parser.create_parse_tree(string, "CommandSequence")
