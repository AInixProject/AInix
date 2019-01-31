from typing import Tuple

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.parse_primitives import AInixParseError
from ainix_common.parsing.stringparser import StringParser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples
from ainix_kernel.specialtypes import allspecials


def get_all_data() -> Tuple[str, str]:
    data = []
    with open('./data/all.nl.filtered') as nlf:
        with open('./data/all.cm.filtered') as cmf:
            for nl, cm in zip(nlf, cmf):
                data.append((nl.strip(), cm.strip()))
    return data


def get_a_tc() -> TypeContext:
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


def stuff(data: Tuple[str, str]):
    tc = get_a_tc()
    parser = StringParser(tc)
    parsable_data = []
    for nl, cm in data:
        try:
            ast = parser.create_parse_tree(cm, "CommandSequence")
            print("success ", (cm, nl))
            parsable_data.append((nl, cm))
        except AInixParseError as e:
            pass
    print(len(parsable_data))


if __name__ == "__main__":
    data = get_all_data()
    stuff(data)
