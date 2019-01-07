from ainix_common.parsing import loader
from ainix_common.parsing.typecontext import TypeContext

LOAD_PATH_ROOT = "ainix_common/tests/toy_contexts_files/"


def get_toy_strings_context() -> TypeContext:
    context = TypeContext()
    loader.load_path(f"builtin_types/generic_parsers.ainix.yaml", context, up_search_limit=4)
    loader.load_path(f"{LOAD_PATH_ROOT}/twostr.ainix.yaml", context, up_search_limit=4)
    context.finalize_data()
    return context
