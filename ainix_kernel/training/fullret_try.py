from argparse import ArgumentParser

import math

from ainix_common.parsing.stringparser import AstUnparser
from ainix_kernel.indexing.examplestore import DataSplits, DEFAULT_SPLITS
from ainix_kernel.models.Fullretrieval.fullretmodel import full_ret_from_example_store, \
    REPLACEMENT_SAMPLES
from ainix_kernel.training.evaluate import EvaluateLogger, print_ast_eval_log
from ainix_kernel.training.trainer import TypeTranslateCFTrainer, get_examples, cookie_monster_path
import os


def train_the_thing(splits = DEFAULT_SPLITS, use_rand_seed = False,
                    replace_samples = REPLACEMENT_SAMPLES,
                    enc_name: str = "CM"):
    pretrained_checkpoint_path = cookie_monster_path()
    type_context, index, replacers, loader = get_examples(splits, use_rand_seed)
    print(f"count {len(list(index.get_all_x_values((DataSplits.VALIDATION,))))}")
    model = full_ret_from_example_store(index, replacers, pretrained_checkpoint_path,
                                        replace_samples, enc_name)
    return model, index, replacers, type_context, loader


if __name__ == "__main__":
    argparer = ArgumentParser()
    default_split_train = DEFAULT_SPLITS[0]
    assert default_split_train[1] == DataSplits.TRAIN
    argparer.add_argument("--train_percent", type=float, default=default_split_train[0]*100)
    argparer.add_argument("--randomize_seed", action='store_true')
    argparer.add_argument("--nointeractive", action='store_true')
    argparer.add_argument("--eval_replace_samples", type=int, default=5)
    argparer.add_argument("--replace_samples", type=int, default=REPLACEMENT_SAMPLES)
    argparer.add_argument("--encoder_name", type=str, default="CM")
    args = argparer.parse_args()
    train_frac = args.train_percent / 100.0
    split_proportions = ((train_frac, DataSplits.TRAIN), (1-train_frac, DataSplits.VALIDATION))

    model, index, replacers, type_context, loader = train_the_thing(
        split_proportions, args.randomize_seed, args.replace_samples, args.encoder_name)
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
