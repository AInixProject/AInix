from ainix_common.parsing.typecontext import *

def test_builtin_singletypeimplparser():
    type_context = TypeContext()
    new_type = AInixType(type_context, "fooType", SINGLE_TYPE_IMPL_BUILTIN)
    assert new_type.default_type_parser._parse_function.__name__ == \
        ainix_common.parsing.parse_primitives.SingleTypeImplParserFunc.__name__


def test_default_fill():
    type_context = TypeContext()
    new_type = AInixType(type_context, "fooType")
    single_impl = AInixObject(type_context, "object", "fooType")
    assert new_type.default_type_parser_name is None
    assert single_impl.preferred_object_parser_name is None
    type_context.fill_default_parsers()
    assert new_type.default_type_parser._parse_function.__name__ == \
        ainix_common.parsing.parse_primitives.SingleTypeImplParserFunc.__name__
    assert single_impl.preferred_object_parser._parse_function.__name__ == \
        ainix_common.parsing.parse_primitives.NoArgsObjectParseFunc.__name__

