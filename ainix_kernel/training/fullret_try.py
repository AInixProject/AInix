from argparse import ArgumentParser

import math

from ainix_common.parsing.stringparser import AstUnparser
from ainix_kernel.indexing.examplestore import DataSplits, DEFAULT_SPLITS, SPLIT_PROPORTIONS_TYPE
from ainix_kernel.models.Fullretrieval.fullretmodel import full_ret_from_example_store
from ainix_kernel.training.evaluate import EvaluateLogger, print_ast_eval_log
from ainix_kernel.training.trainer import TypeTranslateCFTrainer, get_examples
import os


def train_the_thing(
    splits: SPLIT_PROPORTIONS_TYPE = DEFAULT_SPLITS,
    use_rand_seed: bool = False
):
    # bad hacks method thing
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pretrained_checkpoint_path = f"{dir_path}/../../checkpoints/" \
        f"lmchkp_192conv2rnntrans_3epochs_total_3.21_ns0.44_lm2.77"
    #   "lmchkp_30epoch2rnn_merge_toks_total_2.922_ns0.424_lm2.4973.pt"


    type_context, index, replacers, loader = get_examples(splits, use_rand_seed)
    print(f"count {len(list(index.get_all_x_values((DataSplits.VALIDATION,))))}")
    model = full_ret_from_example_store(index, replacers, pretrained_checkpoint_path)
    return model, index, replacers, type_context, loader


if __name__ == "__main__":
    argparer = ArgumentParser()
    default_split_train = DEFAULT_SPLITS[0]
    assert default_split_train[1] == DataSplits.TRAIN
    argparer.add_argument("--train_percent", type=float, default=default_split_train[0]*100)
    argparer.add_argument("--randomize_seed", action='store_true')
    argparer.add_argument("--nointeractive", action='store_true')
    argparer.add_argument("--eval_replace_samples", type=int, default=5)
    args = argparer.parse_args()
    train_frac = args.train_percent / 100.0
    split_proportions = ((train_frac, DataSplits.TRAIN), (1-train_frac, DataSplits.VALIDATION))

    model, index, replacers, type_context, loader = train_the_thing(
        split_proportions, args.randomize_seed)
    unparser = AstUnparser(type_context, model.get_string_tokenizer())

    tran_trainer = TypeTranslateCFTrainer(model, index, replacer=replacers, loader=loader)
    logger = EvaluateLogger()
    tran_trainer.evaluate(logger, dump_each=True, num_replace_samples=args.eval_replace_samples)
    print_ast_eval_log(logger)

    if not args.nointeractive:
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
