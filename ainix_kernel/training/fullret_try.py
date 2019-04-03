import math

from ainix_common.parsing.stringparser import AstUnparser
from ainix_kernel.indexing.examplestore import DataSplits
from ainix_kernel.models.Fullretrieval.fullretmodel import full_ret_from_example_store
from ainix_kernel.training.evaluate import EvaluateLogger, print_ast_eval_log
from ainix_kernel.training.trainer import TypeTranslateCFTrainer, get_examples
import os


def train_the_thing():
    # bad hacks method thing
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pretrained_checkpoint_path = f"{dir_path}/../../checkpoints/" \
        f"lmchkp_192conv2rnntrans_3epochs_total_3.21_ns0.44_lm2.77"
    #                             "lmchkp_30epoch2rnn_merge_toks_total_2.922_ns0.424_lm2.4973.pt"
    type_context, index, replacers, loader = get_examples()
    print(f"count {len(list(index.get_all_examples((DataSplits.VALIDATION,))))}")
    model = full_ret_from_example_store(index, replacers, pretrained_checkpoint_path)
    return model, index, replacers, type_context, loader


if __name__ == "__main__":
    model, index, replacers, type_context, loader = train_the_thing()
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
