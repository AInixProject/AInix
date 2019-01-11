import pytest
from ainix_common.parsing.ast_components import *
from ainix_common.parsing import loader, parse_primitives
from ainix_common.parsing.model_specific.tokenizers import SpaceTokenizer
from ainix_common.parsing.parse_primitives import TypeParser, ArgParseDelegation, TypeParserRun, \
    TypeParserResult, AInixParseError, ParseDelegationReturnMetadata, ObjectParser
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import TypeContext, AInixArgument, AInixObject, AInixType, \
    OPTIONAL_ARGUMENT_NEXT_ARG_NAME
from ainix_common.parsing.grammar_lang import create_object_parser_from_grammar
from ainix_common.tests.toy_contexts import get_toy_strings_context

BUILTIN_TYPES_PATH = "../../builtin_types"


@pytest.fixture(scope="function")
def type_context():
    context = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/command.ainix.yaml", context)
    context.finalize_data()
    return context


@pytest.fixture(scope="function")
def numbers_type_context():
    type_context = TypeContext()
    loader.load_path(f"builtin_types/generic_parsers.ainix.yaml", type_context, up_search_limit=3)
    loader.load_path(f"builtin_types/numbers.ainix.yaml", type_context, up_search_limit=3)
    type_context.finalize_data()
    return type_context


def test_end_to_end_parse1(type_context):
    aArg = AInixArgument(type_context, "a", None, arg_data={"short_name": "a"},
                         parent_object_name="FooProgram")
    bArg = AInixArgument(type_context, "b", None, arg_data={"short_name": "b"},
                         parent_object_name="BarProgram")
    foo = AInixObject(
        type_context, "FooProgram", "Program",
        [aArg, bArg],
        type_data={"invoke_name": "foo"})
    bar = AInixObject(
        type_context, "BarProgram", "Program",
        [AInixArgument(type_context, "b", None, arg_data={"short_name": "b"},
                       parent_object_name="sdf")],
        type_data={"invoke_name": "bar"})
    cmdSeqType = type_context.get_type_by_name("CommandSequence")
    parser = StringParser(type_context)
    result = parser.create_parse_tree("foo -a", cmdSeqType.name)
    assert result.type_to_choose == cmdSeqType
    assert result.choice.implementation == type_context.get_object_by_name("CommandSequenceObj")
    compoundOp: ObjectChoiceNode = result.choice._arg_name_to_node['CompoundOp']
    assert is_obj_choice_a_not_present_node(compoundOp)
    programArg: ObjectChoiceNode = result.choice._arg_name_to_node['ProgramArg']
    assert programArg.type_to_choose == type_context.get_type_by_name("Program")
    assert programArg.choice.implementation == foo
    a_choice: ObjectChoiceNode = programArg.choice._arg_name_to_node[aArg.name]
    assert is_obj_choice_a_present_node(a_choice)
    b_choice: ObjectChoiceNode = programArg.choice._arg_name_to_node[bArg.name]
    assert is_obj_choice_a_not_present_node(b_choice)


def test_no_arg():
    tc = TypeContext()
    AInixType(tc, "FooType")
    AInixObject(tc, "FooO", "FooType")
    tc.finalize_data()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("a", "FooType")


def test_no_arg_delegate():
    tc = TypeContext()
    def del_parse(run: TypeParserRun, string: str, result: TypeParserResult):
        impls = run.all_type_implementations
        assert len(impls) == 1
        d = yield run.delegate_parse_implementation(impls[0], (0, len(string)))
        result.accept_delegation(d)
    p = TypeParser(tc, "del_parser", del_parse)
    AInixType(tc, "FooType", "del_parser")
    o = AInixObject(tc, "FooO", "FooType")
    tc.finalize_data()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("a", "FooType")
    assert ast.next_node_not_copy.implementation == o


def test_no_arg_max_munch():
    tc = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/generic_parsers.ainix.yaml", tc)
    AInixType(tc, "FooType", "max_munch_type_parser")
    o = AInixObject(tc, "FooO", "FooType")
    tc.finalize_data()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("a", "FooType")
    assert ast.next_node_not_copy.implementation == o


@pytest.fixture(scope="function")
def toy_string_context_optional() -> TypeContext:
    tc = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/generic_parsers.ainix.yaml", tc)
    foo_string = AInixType(tc, "FooString")
    foo_string_obj = AInixObject(tc, "foo_string_obj", "FooString",
                                 children=[AInixArgument(tc, "CurWord", "FooWord",
                                                         required=True, parent_object_name="sdf"),
                                           AInixArgument(tc, "Nstr", "FooString", required=False,
                                                         parent_object_name="sdfab")],
                                 preferred_object_parser_name="foo_string_parser")
    foo_string_parser = create_object_parser_from_grammar(
        tc, "foo_string_parser", r"CurWord Nstr?")
    foo_word = AInixType(tc, "FooWord", "max_munch_type_parser")
    objects = [AInixObject(tc, "FooWordOf" + name, "FooWord",
                           preferred_object_parser_name=create_object_parser_from_grammar(
                                tc, f"ParserOf{name}", f'"{name}"'
                           ).name)
               for name in ("a", "bee", "c")]
    tc.finalize_data()
    return tc


def test_parse_with_delegations(toy_string_context_optional):
    """A test where one of the object parsers (a grammar parser) tries to
    delegate its parseing"""
    parser = StringParser(toy_string_context_optional)
    ast = parser.create_parse_tree("a", "FooString")
    foo_string_obj = ast.next_node
    assert foo_string_obj.implementation.name == "foo_string_obj"
    cur_word = foo_string_obj.get_choice_node_for_arg("CurWord")
    assert cur_word.get_chosen_impl_name() == "FooWordOfa"
    next_str = foo_string_obj.get_choice_node_for_arg("Nstr")
    print(next_str)
    assert is_obj_choice_a_not_present_node(next_str)


def test_parse_arg_delegation(toy_string_context_optional):
    parser = StringParser(toy_string_context_optional)
    foo_string_obj = toy_string_context_optional.get_object_by_name("foo_string_obj")
    metadata, node = parser._delegate_object_arg_parse(ArgParseDelegation(
        foo_string_obj.get_arg_by_name("CurWord"),
        foo_string_obj,
        "abee",
        (0, 4)
    ))
    assert metadata.parse_success
    assert metadata.remaining_right_starti == 1
    assert metadata.remaining_string == "bee"
    assert node.get_chosen_impl_name() == "FooWordOfa"


def test_parse_arg_delegation2(toy_string_context_optional):
    parser = StringParser(toy_string_context_optional)
    foo_string_obj = toy_string_context_optional.get_object_by_name("foo_string_obj")
    metadata, node = parser._delegate_object_arg_parse(ArgParseDelegation(
        foo_string_obj.get_arg_by_name("Nstr"),
        foo_string_obj,
        "bee",
        (1, 4)
    ))
    assert metadata.parse_success
    assert metadata.remaining_string == ""
    assert metadata.remaining_right_starti == 3
    assert is_obj_choice_a_present_node(node)


def test_parse_with_delegations2(toy_string_context_optional):
    """A test where one of the object parsers (a grammar parser) tries to
    delegate its parseing"""
    parser = StringParser(toy_string_context_optional)
    ast = parser.create_parse_tree("abee", "FooString")
    foo_string_obj = ast.next_node
    assert foo_string_obj.implementation.name == "foo_string_obj"
    cur_word = foo_string_obj.get_choice_node_for_arg("CurWord")
    assert cur_word.get_chosen_impl_name() == "FooWordOfa"
    next_str = foo_string_obj.get_choice_node_for_arg("Nstr")
    print(next_str)
    assert is_obj_choice_a_present_node(next_str)


@pytest.fixture(scope="function")
def numbers_ast_set(numbers_type_context) -> AstObjectChoiceSet:
    root_type_name = "Number"
    root_type = numbers_type_context.get_type_by_name(root_type_name)
    choice_set = AstObjectChoiceSet(root_type, None)
    return choice_set


def test_parse_set_1(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast = parser.create_parse_tree("5", root_type_name)
    numbers_ast_set.add(ast, True, 1, 1)
    assert numbers_ast_set.is_node_known_valid(ast)
    assert not numbers_ast_set.is_node_known_valid(
        parser.create_parse_tree("9", root_type_name))
    assert not numbers_ast_set.is_node_known_valid(
        parser.create_parse_tree("-5", root_type_name))


def test_parse_set_freeze(numbers_type_context, numbers_ast_set):
    root_type_name = "Number"
    parser = StringParser(numbers_type_context)
    ast = parser.create_parse_tree("5", root_type_name)
    numbers_ast_set.add(ast, True, 1, 1)
    numbers_ast_set.freeze()
    with pytest.raises(ValueError):
        numbers_ast_set.add(ast, True, 1, 1)
    real_set = {numbers_ast_set}
    assert numbers_ast_set in real_set


def test_parse_set_2(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("-5", root_type_name)
    print(ast_1.dump_str())
    numbers_ast_set.add(ast_1, True, 1, 1)
    print("---")
    assert numbers_ast_set.is_node_known_valid(ast_1)


def test_parse_set_3(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("-5", root_type_name)
    ast_3 = parser.create_parse_tree("50", root_type_name)
    ast_4 = parser.create_parse_tree("6", root_type_name)
    numbers_ast_set.add(ast_1, True, 1, 1)
    numbers_ast_set.add(ast_2, True, 1, 1)
    numbers_ast_set.add(ast_3, True, 1, 0.2)
    numbers_ast_set.add(ast_4, True, 1, 1)
    assert numbers_ast_set.is_node_known_valid(ast_1)
    assert numbers_ast_set.is_node_known_valid(ast_2)
    assert numbers_ast_set.is_node_known_valid(ast_3)
    assert numbers_ast_set.is_node_known_valid(ast_4)


def test_parse_set_4(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    numbers_ast_set.add(ast_1, False, 1, 1)
    assert not numbers_ast_set.is_node_known_valid(ast_1)


def test_parse_set_5(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("-5", root_type_name)
    numbers_ast_set.add(ast_1, True, 1, 1)
    numbers_ast_set.add(ast_2, False, 1, 1)
    assert numbers_ast_set.is_node_known_valid(ast_1)
    assert not numbers_ast_set.is_node_known_valid(ast_2)


def test_parse_set_6(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("50", root_type_name)
    numbers_ast_set.add(ast_1, True, 1, 1)
    print("---")
    numbers_ast_set.add(ast_2, False, 1, 0.3)
    assert numbers_ast_set.is_node_known_valid(ast_1)
    print(ast_2.dump_str())
    assert not numbers_ast_set.is_node_known_valid(ast_2)


def test_parse_set_7(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("50", root_type_name)
    numbers_ast_set.add(ast_1, False, 1, 1)
    numbers_ast_set.add(ast_2, True, 1, 0.3)
    assert not numbers_ast_set.is_node_known_valid(ast_1)
    assert numbers_ast_set.is_node_known_valid(ast_2)


def test_parse_set_8(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast_1 = parser.create_parse_tree("5", root_type_name)
    ast_2 = parser.create_parse_tree("50", root_type_name)
    ast_3 = parser.create_parse_tree("500", root_type_name)
    numbers_ast_set.add(ast_1, True, 1, 1)
    numbers_ast_set.add(ast_2, False, 1, 0.3)
    numbers_ast_set.add(ast_3, True, 1, 1)
    assert numbers_ast_set.is_node_known_valid(ast_1)
    assert not numbers_ast_set.is_node_known_valid(ast_2)
    assert numbers_ast_set.is_node_known_valid(ast_3)


def test_parse_set_9(numbers_type_context, numbers_ast_set):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast = parser.create_parse_tree("5", root_type_name)
    numbers_ast_set.add(ast, True, 1, 1)
    new_ast = parser.create_parse_tree("5", root_type_name)
    assert numbers_ast_set.is_node_known_valid(new_ast)


def test_set_with_copy():
    tc = TypeContext()
    ft = AInixType(tc, "FT")
    AInixObject(tc, "FO", "FT")
    bo = AInixObject(tc, "BO", "FT")
    ast = ObjectChoiceNode(ft)
    copy = CopyNode(ft, 0, 3)
    ast.set_choice(copy)
    ast.freeze()
    #
    ast_set = AstObjectChoiceSet(ft, None)
    assert not ast_set.copy_is_known_choice()
    ast_set.add(ast, True, 1, 1)
    assert ast_set.copy_is_known_choice()
    assert ast_set.is_node_known_valid(ast)
    #
    other_ast = ObjectChoiceNode(ft, None)
    other_ast.set_choice(ObjectNode(bo))
    assert not ast_set.is_node_known_valid(other_ast)
    ast_set.add(other_ast, True, 1, 1)
    assert ast_set.is_node_known_valid(other_ast)


@pytest.fixture(scope="function")
def simple_p_tc():
    tc = TypeContext()
    loader.load_path(f"{BUILTIN_TYPES_PATH}/generic_parsers.ainix.yaml", tc)
    AInixType(
        type_context=tc,
        name="FooType"
    )
    AInixType(
        type_context=tc,
        name="BarType"
    )
    create_object_parser_from_grammar(tc, "AParser", '"Hello " Arg1')
    AInixObject(
        type_context=tc,
        name="objOfFoo",
        type_name="FooType",
        children=[AInixArgument(tc, "Arg1", "BarType", parent_object_name="objOfFoo")],
        preferred_object_parser_name="AParser"
    )
    create_object_parser_from_grammar(tc, "BParser", '"Bar"')
    AInixObject(
        type_context=tc,
        name="objOfBar",
        type_name="BarType",
        children=[],
        preferred_object_parser_name="BParser"
    )
    tc.finalize_data()
    return tc


def test_unparse_single(simple_p_tc):
    parser = StringParser(simple_p_tc)
    root_type_name = "FooType"
    ast = parser.create_parse_tree("Hello Bar", root_type_name)
    unparser = AstUnparser(simple_p_tc)
    result = unparser.to_string(ast)
    assert result.total_string == "Hello Bar"
    # Test spans
    assert result.node_to_string(ast) == "Hello Bar"
    objectOfFoo = ast.next_node
    assert result.node_to_string(objectOfFoo) == "Hello Bar"
    arg1_is_present = objectOfFoo.get_choice_node_for_arg("Arg1")
    assert result.node_to_string(arg1_is_present) == "Bar"
    arg1_present_o = arg1_is_present.next_node
    assert result.node_to_string(arg1_present_o) == "Bar"
    arg1_typechoice = arg1_present_o.get_choice_node_for_arg(OPTIONAL_ARGUMENT_NEXT_ARG_NAME)
    assert result.node_to_string(arg1_typechoice) == "Bar"
    baro = arg1_typechoice.next_node
    assert result.node_to_string(baro) == "Bar"
    for pointer in ast.depth_first_iter():
        assert pointer.cur_node in result.node_to_span


def test_unparse_double():
    tc = get_toy_strings_context()
    parser = StringParser(tc)
    root_type_name = "ToySimpleStrs"
    ast = parser.create_parse_tree("TWO foo bar", root_type_name)
    unparser = AstUnparser(tc)
    result = unparser.to_string(ast)
    assert result.total_string == "TWO foo bar"
    assert isinstance(ast.next_node, ObjectNode)
    arg1_choice: ObjectChoiceNode = ast.next_node.get_choice_node_for_arg("arg1")
    assert result.node_to_string(arg1_choice) == "foo"
    arg1_ob: ObjectNode = arg1_choice.next_node
    assert result.node_to_string(arg1_ob) == "foo"


def test_unparse_no_arg():
    tc = TypeContext()
    ft = AInixType(tc, "ft")
    bt = AInixType(tc, "bt")
    arg1 = AInixArgument(tc, "arg1", "bt", required=True)
    fo = AInixObject(tc, "fo", "ft", [arg1],
                     preferred_object_parser_name=create_object_parser_from_grammar(
                         tc, "foo_par", '"foo" arg1?'
                     ).name)
    bo = AInixObject(tc, "bo", "bt", [],
                     preferred_object_parser_name=create_object_parser_from_grammar(
                         tc, "bar_par", '"here"'
                     ).name)
    tc.finalize_data()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("foo here", "ft")
    unparser = AstUnparser(tc)
    result = unparser.to_string(ast)
    assert result.total_string == "foohere"
    noargs = ast.next_node_not_copy.get_choice_node_for_arg("arg1")
    assert result.node_to_string(noargs) == "here"
    for pointer in ast.depth_first_iter():
        assert pointer.cur_node in result.node_to_span


def test_unparse_no_arg_no_str():
    tc = TypeContext()
    ft = AInixType(tc, "ft")
    bt = AInixType(tc, "bt")
    arg1 = AInixArgument(tc, "arg1", "bt", required=True)
    fo = AInixObject(tc, "fo", "ft", [arg1],
                     preferred_object_parser_name=create_object_parser_from_grammar(
                         tc, "foo_par", '"foo" arg1?'
                     ).name)
    bo = AInixObject(tc, "bo", "bt", [])
    tc.finalize_data()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("fooasdf", "ft")
    unparser = AstUnparser(tc)
    result = unparser.to_string(ast)
    assert result.total_string == "foo"
    noargs = ast.next_node_not_copy.get_choice_node_for_arg("arg1")
    assert result.node_to_string(noargs) == ""
    for pointer in ast.depth_first_iter():
        assert pointer.cur_node in result.node_to_span


def test_unparse_copy_node():
    tc = TypeContext()
    ft = AInixType(tc, "ft")
    fo = AInixObject(tc, "fo", "ft")
    tc.finalize_data()
    ast = ObjectChoiceNode(ft)
    ast.set_choice(CopyNode(ft, 2, 4))
    ast.freeze()
    unparser = AstUnparser(tc, SpaceTokenizer())
    in_str = "hello there foo bar pop wow"
    result = unparser.to_string(ast, in_str)
    assert result.total_string == "foo bar pop"


def test_unparse_copy_node_in_grammar():
    tc = get_toy_strings_context()
    ast = ObjectChoiceNode(tc.get_type_by_name("ToySimpleStrs"))
    ch1 = ObjectNode(tc.get_object_by_name("two_string"))
    ast.set_choice(ch1)
    toy_meta = tc.get_type_by_name("ToyMetasyntactic")
    a1 = ObjectChoiceNode(toy_meta)
    a1.set_choice(CopyNode(toy_meta, 2, 2))
    ch1.set_arg_value("arg1", a1)
    a2 = ObjectChoiceNode(toy_meta)
    a2.set_choice(CopyNode(toy_meta, 3, 3))
    ch1.set_arg_value("arg2", a2)
    ast.freeze()
    unparser = AstUnparser(tc, SpaceTokenizer())
    in_str = "hello there foo bar pop wow"
    result = unparser.to_string(ast, in_str)
    assert result.total_string == "TWO foo bar"


def test_unparse_optional_arg_copy():
    tc = TypeContext()
    ft = AInixType(tc, "ft")
    arg1 = AInixArgument(tc, "arg1", "ft", required=False, parent_object_name="fo")
    fo = AInixObject(tc, "fo", "ft", [arg1],
                     preferred_object_parser_name=create_object_parser_from_grammar(
                         tc, "masfoo_parser", '"foo" arg1?'
                     ).name)
    tc.finalize_data()
    ast = ObjectChoiceNode(ft)
    co = ObjectNode(fo)
    ast.set_choice(co)
    a1 = ObjectChoiceNode(ft)
    a1.set_choice(CopyNode(ft, 2, 2))
    co.set_arg_value("arg1", a1)
    ast.freeze()

    unparser = AstUnparser(tc, SpaceTokenizer())
    in_str = "we like foo foo things"
    result = unparser.to_string(ast, in_str)
    assert result.total_string == "foofoo"


def test_unparse_optional_arg():
    tc = TypeContext()
    ft = AInixType(tc, "ft")
    arg1 = AInixArgument(tc, "arg1", "ft", required=False, parent_object_name="fo")
    fo = AInixObject(tc, "fo", "ft", [arg1],
                     preferred_object_parser_name=create_object_parser_from_grammar(
                        tc, "masfoo_parser", '"foo" arg1?'
                     ).name)
    tc.finalize_data()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("foo", "ft")
    unparser = AstUnparser(tc)
    result = unparser.to_string(ast)
    assert result.total_string == "foo"
    argnode = ast.next_node_not_copy.get_choice_node_for_arg("arg1")
    #assert result.node_to_string(argnode) == ""
    # This currently doesn't work and it is not clear whether it even should
    #for pointer in ast.depth_first_iter():
    #    assert pointer.cur_node in result.node_to_span


def test_unparse_optional_arg_cust_parser():
    tc = TypeContext()
    ft = AInixType(tc, "ft")
    arg1 = AInixArgument(tc, "arg1", "ft", required=False, parent_object_name="sdf")
    def cust_foo_func(
        run: parse_primitives.ObjectParserRun,
        string: str,
        result: parse_primitives.ObjectParserResult
    ):
        if not string.startswith("foo"):
            raise AInixParseError("That's not a foo string!")
        deleg = yield run.left_fill_arg(arg1, (len("foo"), len(string)))
        if deleg.parse_success:
            result.accept_delegation(deleg)
    def unparser(
        arg_map: parse_primitives.ObjectNodeArgMap,
        result: parse_primitives.ObjectToStringResult
    ):
        result.add_string("foo")
        result.add_arg_tostring(arg1)

    fo = AInixObject(tc, "fo", "ft", [arg1],
                     ObjectParser(tc, 'pname', cust_foo_func, unparser).name)
    tc.finalize_data()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("foo", "ft")
    unparser = AstUnparser(tc)
    result = unparser.to_string(ast)
    assert result.total_string == "foo"
    argnode = ast.next_node_not_copy.get_choice_node_for_arg("arg1")


def test_simple_num_unparse(numbers_type_context):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast = parser.create_parse_tree("5", root_type_name)
    unparse = AstUnparser(numbers_type_context)
    result = unparse.to_string(ast)
    assert result.total_string == "5"


def test_num_unparse2(numbers_type_context):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast = parser.create_parse_tree("51", root_type_name)
    unparse = AstUnparser(numbers_type_context)
    result = unparse.to_string(ast)
    assert result.total_string == "51"


def test_num_unparse3(numbers_type_context):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast = parser.create_parse_tree("51.0", root_type_name)
    unparse = AstUnparser(numbers_type_context)
    result = unparse.to_string(ast)
    assert result.total_string == "51.0"


def test_num_unparse4(numbers_type_context):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast = parser.create_parse_tree("-51.0", root_type_name)
    unparse = AstUnparser(numbers_type_context)
    result = unparse.to_string(ast)
    assert result.total_string == "-51.0"


def test_num_unparse5(numbers_type_context):
    parser = StringParser(numbers_type_context)
    root_type_name = "Number"
    ast = parser.create_parse_tree("5.4e8", root_type_name)
    unparse = AstUnparser(numbers_type_context)
    result = unparse.to_string(ast)
    assert result.total_string == "5.4e8"


def test_unparse_second_arg():
    tc = get_toy_strings_context()
    parser = StringParser(tc)
    unparser = AstUnparser(tc)
    string = "TWO foo bar"
    ast = parser.create_parse_tree(string, "ToySimpleStrs")
    result = unparser.to_string(ast)
    assert result.total_string == string
    arg2 = ast.next_node_not_copy.get_choice_node_for_arg("arg2")
    assert result.node_to_span[arg2] == (8, 11)
    assert result.node_to_string(arg2) == "bar"
