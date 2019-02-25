import math
import random
from typing import Tuple, Optional

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.stringparser import AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.models.Fullretrieval.fullretmodel import full_ret_from_example_store
from ainix_kernel.specialtypes import allspecials
from ainix_kernel.training.augmenting.replacers import get_all_replacers
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples


if __name__ == "__main__":
    pretrained_checkpoint_path = "../../checkpoints/" \
                                 "lmchkp_iter152k_200_2rnn_total3.29_ns0.47_lm2.82.pt"

    type_context = TypeContext()
    loader = TypeContextDataLoader(type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(type_context)

    for f in ALL_EXAMPLE_NAMES:
        loader.load_path(f"builtin_types/{f}.ainix.yaml")
    type_context.finalize_data()

    index = load_all_examples(type_context)
    #index = load_tellia_examples(type_context)

    print("num docs", index.backend.index.doc_count())
    replacers = get_all_replacers()

    model = full_ret_from_example_store(index, replacers, pretrained_checkpoint_path)
    unparser = AstUnparser(type_context, model.get_string_tokenizer())


    while True:
        q = input("Query: ")
        ast, metad = model.predict(q, "CommandSequence", False)
        unparse_result = unparser.to_string(ast, q)
        print(unparse_result.total_string)
        print(math.exp(sum(metad.log_confidences)))
        print(index.get_example_by_id(
            metad.example_retrieve_explanations[0].reference_example_ids[0]))
