from ainix_common.parsing import parse_primitives
from ainix_common.parsing.parse_primitives import ObjectParser, TypeParser
from ainix_common.parsing.typecontext import TypeContext, AInixType, AInixObject, AInixArgument
import pygtrie
from typing import List

WORD_PART_TERMINAL_NAME = "generic_word_part_terminal"
WORD_PART_TYPE_PARSER_NAME = "word_part_type_parser"


def create_generic_strings(type_context: TypeContext):
    _create_root_types(type_context)


def _get_all_symbols() -> List[str]:
    #first_symbls = [chr(c) for c in range(32, 65)]
    #middle_symbls = [chr(c) for c in range(91, 97)]
    #end_symbls = [chr(c) for c in range(123, 127)]
    #return first_symbls + middle_symbls + end_symbls
    # don't actually return everything because then will consume everything
    return ["_", "$"]


def _get_all_letters() -> List[str]:
    return [chr(c) for c in range(97, 123)]


def _create_root_types(type_context: TypeContext):
    """Creates the underlying types for handling generid strings"""
    AInixType(type_context, "GenericWordPart", WORD_PART_TYPE_PARSER_NAME)
    AInixObject(type_context, WORD_PART_TERMINAL_NAME, "GenericWordPart")

    AInixType(type_context, "GenericWordPartModifier")
    AInixObject(type_context, "unmodified_lower", "GenericWordPartModifier")
    AInixObject(type_context, "first_letter_upper", "GenericWordPartModifier")
    AInixObject(type_context, "all_letters_upper", "GenericWordPartModifier")


def _create_word_part_obj(tc: TypeContext, symb: str, allow_modifier: bool) -> AInixObject:
    """Creates an individual word part object with the proper parsers"""
    children = [AInixArgument(tc, "next_part", "GenericWordPart")]
    if allow_modifier:
        children += [AInixArgument(tc, "modifier", "GenericWordPartModifier")]

    def parser(
        run: parse_primitives.ObjectParserRun,
        string: str,
        result: parse_primitives.ObjectParserResult
    ):
        if not string.startswith(symb):
            raise parse_primitives.AInixParseError(
                f"Expected string {string} to start with {symb}")
        if allow_modifier:
            mod_arg = run.all_arguments[1]
            result.set_arg_present(mod_arg.name, 0, len(symb))

    def unparser(
        arg_map: parse_primitives.ObjectNodeArgMap,
        result: parse_primitives.ObjectToStringResult
    ):
        result.add_string(symb)
    p = ObjectParser(tc, f"parser_for_word_part_{symb}", parser, unparser)
    part = AInixObject(tc, f"word_part_{symb}", "GenericWordPart", children, p.name)
    return part


def _create_all_word_parts(tc: TypeContext):
    all_part_calls = [(tc, symb, False) for symb in _get_all_symbols()] + \
                     [(tc, symb, True) for symb in _get_all_letters()]
    symb_trie = pygtrie.CharTrie()
    for tc, symb, allow_mod in all_part_calls:
        new_part = _create_word_part_obj(tc, symb, allow_mod)
        symb_trie[symb] = new_part
        if allow_mod:
            symb_trie[symb.upper()] = new_part
            first_letter_upper_version = symb
            first_letter_upper_version[0] = first_letter_upper_version[0].upper()
            symb_trie[first_letter_upper_version] = new_part
    symb_trie[""] = tc.get_object_by_name(WORD_PART_TERMINAL_NAME)

    def word_parser_func(
        run: parse_primitives.TypeParserRun,
        string: str,
        result: parse_primitives.TypeParserResult
    ):
        result.set_valid_implementation(symb_trie.longest_prefix(string).value)
    TypeParser(tc, WORD_PART_TYPE_PARSER_NAME, word_parser_func)


if __name__ == "__main__":
    print(_get_all_letters())
