import math
import random
from typing import Tuple, Optional

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.stringparser import AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import DataSplits
from ainix_kernel.models.Fullretrieval.fullretmodel import full_ret_from_example_store
from ainix_kernel.specialtypes import allspecials
from ainix_kernel.training.augmenting.replacers import get_all_replacers
from ainix_kernel.training.evaluate import EvaluateLogger, print_ast_eval_log
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples, \
    load_tellia_examples
from ainix_kernel.training.trainer import TypeTranslateCFTrainer

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

    print("num docs", index.get_doc_count())
    print("num train", len(list(index.get_all_examples((DataSplits.TRAIN, )))))

    replacers = get_all_replacers()

    model = full_ret_from_example_store(index, replacers, pretrained_checkpoint_path)
    unparser = AstUnparser(type_context, model.get_string_tokenizer())

    tran_trainer = TypeTranslateCFTrainer(model, index, replacer=replacers, loader=loader)
    logger = EvaluateLogger()
    tran_trainer.evaluate(logger, dump_each=True, num_replace_samples=5)
    print_ast_eval_log(logger)

    while True:
        q = input("Query: ")
        ast, metad = model.predict(q, "CommandSequence", True)
        unparse_result = unparser.to_string(ast, q)
        print(unparse_result.total_string)
        print(math.exp(sum(metad.log_confidences)))
        retr_explan = metad.example_retrieve_explanations[0]
        for sim, example_id in zip(
            retr_explan.reference_confidence,
            retr_explan.reference_example_ids
        ):
            print(math.exp(sim), index.get_example_by_id(example_id))
