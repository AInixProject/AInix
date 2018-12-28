"""Provides the types, objects, and parsers for parsing 'generic strings'
which are basically just a string of usually natural language words,
but poteneitally might just be other letters and words"""
from ainix_common.parsing import parse_primitives
from ainix_common.parsing.parse_primitives import ObjectParser, TypeParser
from ainix_common.parsing.typecontext import TypeContext, AInixType, AInixObject, AInixArgument
import pygtrie
from typing import List, Tuple

WORD_PART_TYPE_NAME = "GenericWordPart"
WORD_PART_TERMINAL_NAME = "generic_word_part_terminal"
WORD_PART_TYPE_PARSER_NAME = "word_part_type_parser"
MODIFIER_LOWER_NAME = "unmodified_lower"
MODIFIER_FIRST_CAP_NAME = "modifier_first_cap"
MODIFIER_ALL_UPPER = "all_cap_modifier"
WORD_PART_MODIFIER_ARG_NAME = "modifier"
WORD_PART_NEXT_ARG_NAME = "next_part"


def create_generic_strings(type_context: TypeContext):
    """Main public interface for creating the appropriate types inside context"""
    _create_root_types(type_context)
    all_part_strs = [(symb, False) for symb in _get_all_symbols()] + \
                    [(symb, True) for symb in _get_all_letters()]
    _create_all_word_parts(type_context, all_part_strs)


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
    AInixType(type_context, WORD_PART_TYPE_NAME, WORD_PART_TYPE_PARSER_NAME)
    AInixObject(type_context, WORD_PART_TERMINAL_NAME, WORD_PART_TYPE_NAME)
    _create_modifier_types(type_context)


def _create_modifier_types(type_context: TypeContext):
    modifer_type_parser_name = "str_modifier_type_parser"
    AInixType(type_context, "GenericWordPartModifier", modifer_type_parser_name)
    lower = AInixObject(type_context, MODIFIER_LOWER_NAME, "GenericWordPartModifier")
    first_cap = AInixObject(type_context, MODIFIER_FIRST_CAP_NAME, "GenericWordPartModifier")
    all_upper = AInixObject(type_context, MODIFIER_ALL_UPPER, "GenericWordPartModifier")

    def mod_parser_func(
        run: parse_primitives.TypeParserRun,
        string: str,
        result: parse_primitives.TypeParserResult
    ) -> None:
        result.set_next_slice(len(string), len(string))
        if string.isupper():
            result.set_valid_implementation(all_upper)
        elif string.islower():
            result.set_valid_implementation(lower)
        elif string[0].isupper() and string[1:].islower():
            result.set_valid_implementation(first_cap)
        else:
            raise parse_primitives.AInixParseError(
                f"String {string} did match an expected modifier")
    TypeParser(type_context, modifer_type_parser_name, mod_parser_func)


def _name_for_word_part(part_string: str):
    return f"word_part_{part_string}"


def _create_word_part_obj(tc: TypeContext, symb: str, allow_modifier: bool) -> AInixObject:
    """Creates an individual word part object with the proper parsers"""
    if allow_modifier:
        children = [AInixArgument(tc, WORD_PART_MODIFIER_ARG_NAME, "GenericWordPartModifier", required=True)]
    else:
        children = []
    children += [AInixArgument(tc, WORD_PART_NEXT_ARG_NAME, "GenericWordPart", required=True)]

    def parser(
        run: parse_primitives.ObjectParserRun,
        string: str,
        result: parse_primitives.ObjectParserResult
    ):
        if not string.startswith(symb):
            raise parse_primitives.AInixParseError(
                f"Expected string {string} to start with {symb}")
        if allow_modifier:
            mod_arg = run.all_arguments[0]
            next_arg = run.all_arguments[1]
            deleg = yield run.left_fill_arg(mod_arg, (0, len(symb)))
            result.accept_delegation(deleg)
        else:
            next_arg = run.all_arguments[0]
        deleg = yield run.left_fill_arg(next_arg, (len(symb), len(string)))
        result.accept_delegation(deleg)

    def unparser(
        arg_map: parse_primitives.ObjectNodeArgMap,
        result: parse_primitives.ObjectToStringResult
    ):
        result.add_string(symb)
    p = ObjectParser(tc, f"parser_for_word_part_{symb}", parser, unparser)
    part = AInixObject(tc, _name_for_word_part(symb), WORD_PART_TYPE_NAME, children, p.name)
    return part


def _create_all_word_parts(
    tc: TypeContext,
    word_part_strs: List[Tuple[str, bool]]
):
    symb_trie = pygtrie.CharTrie()
    for symb, allow_mod in word_part_strs:
        new_part = _create_word_part_obj(tc, symb, allow_mod)
        symb_trie[symb] = new_part
        if allow_mod:
            symb_trie[symb.upper()] = new_part
            first_letter_upper_version = symb[0].upper() + symb[1:]
            symb_trie[first_letter_upper_version] = new_part
    symb_trie[""] = tc.get_object_by_name(WORD_PART_TERMINAL_NAME)

    def word_parser_func(
        run: parse_primitives.TypeParserRun,
        string: str,
        result: parse_primitives.TypeParserResult
    ):
        result.set_valid_implementation(symb_trie.longest_prefix(string).value)
        result.set_next_slice(0, len(string))
    TypeParser(tc, WORD_PART_TYPE_PARSER_NAME, word_parser_func)


if __name__ == "__main__":
    print(_get_all_letters())
