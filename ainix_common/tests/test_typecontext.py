from typecontext import *
import parse_primitives

def test_builtin_singletypeimplparser():
    type_context = TypeContext()
    new_type = AInixType(type_context, "fooType", SINGLE_TYPE_IMPL_BUILTIN)
    assert new_type.default_type_parser._parse_function == \
        parse_primitives.SingleTypeImplParserFunc
