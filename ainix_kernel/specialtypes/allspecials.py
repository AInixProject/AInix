from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.specialtypes.generic_strings import create_generic_strings


def load_all_special_types(type_context: TypeContext):
    create_generic_strings(type_context)