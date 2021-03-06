import random
from argparse import ArgumentParser

import torch
from typing import Tuple, Generator, List, Set, Optional

from ainix_common.parsing.copy_tools import add_copies_to_ast_set
from ainix_common.parsing.model_specific.tokenizers import StringTokenizer, StringTokensMetadata
from ainix_kernel.indexing.examplestore import ExamplesStore, DataSplits, XValue, YValue, \
    SPLIT_PROPORTIONS_TYPE, DEFAULT_SPLITS
from ainix_kernel.models.model_types import StringTypeTranslateCF, ModelCantPredictException, \
    ModelSafePredictError
from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_kernel.training.augmenting.replacers import Replacer, get_all_replacers, \
    ReplacementSampling
from ainix_kernel.training.evaluate import AstEvaluation, EvaluateLogger, print_ast_eval_log
import more_itertools
from ainix_kernel.specialtypes import allspecials
from ainix_kernel.training.model_specific_training import update_latent_store_from_examples
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples, \
    load_tellina_examples, load_all_and_tellina
from ainix_kernel.util.sampling import WeightedRandomChooser
from ainix_kernel.util.serialization import serialize
from tqdm import tqdm
from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.typecontext import AInixType
from ainix_common.parsing.typecontext import TypeContext, AInixType
import os


def cookie_monster_path():
    # bad hacks method thing
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return f"{dir_path}/../../checkpoints/" \
        f"lmchkp_192conv2rnntrans_3epochs_total_3.21_ns0.44_lm2.77"
    #  "lmchkp_30epoch2rnn_merge_toks_total_2.922_ns0.424_lm2.4973.pt"



class TypeTranslateCFTrainer:
    def __init__(
        self,
        model: StringTypeTranslateCF,
        example_store: ExamplesStore,
        batch_size: int = 1,
        replacer: Replacer = None,
        loader: TypeContextDataLoader = None
    ):
        self.model = model
        self.example_store = example_store
        self.type_context = example_store.type_context
        self.string_parser = StringParser(self.type_context)
        self.batch_size = batch_size
        self.str_tokenizer = self.model.get_string_tokenizer()
        self.unparser = AstUnparser(self.type_context, self.str_tokenizer)
        self.replacer = replacer
        self.loader = loader
        if self.replacer is None:
            self.replacer = Replacer([])

    def _train_one_epoch(self, which_epoch_on: int):
        train_splits = (DataSplits.TRAIN,)
        train_count = sum(1 for _ in self.example_store.get_all_x_values(train_splits))
        single_examples_iter = self.data_pair_iterate(train_splits)
        batches_iter = more_itertools.chunked(single_examples_iter, self.batch_size)
        loss = torch.tensor(0.0)
        examples_seen = 0
        with tqdm(total=train_count, unit='Examples', miniters=train_count/5) as pbar:
            for batch in batches_iter:
                batch_as_query = [(replaced_x, y_ast_set, this_example_ast, example.id) for
                                  example, replaced_x, y_ast_set, this_example_ast, _, _ in batch]
                loss += float(self.model.train_batch(batch_as_query))
                examples_seen += len(batch)
                pbar.update(len(batch))
        self.model.end_train_epoch()
        return loss / examples_seen

    def train(
            self, epochs: int, eval_every_n_epochs: int = None,
            intermitted_save_path = None, dump_each_in_eval: bool = True,
            intermitted_repl_samples: int = 10):
        self.model.start_train_session()
        for epoch in tqdm(range(epochs), unit="Epochs"):
            print()
            print(f"Start epoch {epoch}")
            loss = self._train_one_epoch(epoch)
            print(f"\nEpoch {epoch} complete. Loss {loss}")
            if hasattr(self.model, "plz_train_this_latent_store_thanks"):
                # TODO wasdfahwerdfgv I should sleep
                # (yeah, even with sleep to lazy to fix this crappy interface. It works for now...)
                latent_store = self.model.plz_train_this_latent_store_thanks()
                if latent_store:
                    print("updateding the latent store 🦔")
                    update_latent_store_from_examples(self.model, latent_store, self.example_store,
                                                      self.replacer, self.string_parser,
                                                      (DataSplits.TRAIN,), self.unparser,
                                                      self.str_tokenizer)
            if eval_every_n_epochs and \
                    epoch + 1 != epochs and \
                    epoch % eval_every_n_epochs == 0 and \
                    epoch > 0:
                print("Pausing to do an eval")
                logger = EvaluateLogger()
                self.evaluate(logger, dump_each=dump_each_in_eval,
                              num_replace_samples=intermitted_repl_samples)
                print_ast_eval_log(logger)
                if intermitted_save_path:
                    if self.loader is None:
                        raise ValueError("Must be given loader to serialize")
                    s_path = f"{intermitted_save_path}_epoch{epoch}_exactmatch_" + \
                             f"{logger.stats['ExactMatch'].percent_true_str}"
                    print(f"serializing to {s_path}")
                    serialize(self.model, self.loader,
                              s_path,
                              eval_results=logger,
                              trained_epochs=epoch)

        self.model.end_train_session()

    def evaluate(
        self,
        logger: EvaluateLogger,
        filter_splits: Optional[Tuple[DataSplits]] = (DataSplits.VALIDATION,),
        dump_each: bool = False,
        num_replace_samples = 1
    ):
        self.model.set_in_eval_mode()
        # Kinda hacky approximation of sampling replacements multiple times
        # just iterate overy everything multiple times
        dups = [list(self.data_pair_iterate(filter_splits)) for _ in range(num_replace_samples)]
        dups = flatten_list(dups)
        dups.sort(key=lambda d: d[0].id)

        last_x = None
        last_eval_result = None
        last_example_id = None
        for data in dups:
            example, replaced_x_query, y_ast_set, this_example_ast, y_texts, rsample = data
            if last_x == replaced_x_query:
                logger.add_evaluation(last_eval_result)
                continue
            parse_exception = None
            try:
                prediction, metad = self.model.predict(
                    replaced_x_query, example.get_y_type(self.example_store), True)
            except ModelCantPredictException as e:
                prediction = None
                parse_exception = e
            except ModelSafePredictError as e:
                prediction = None
                parse_exception = e
            #print("predict", prediction, "expect", y_ast_set, "ytext", y_texts,
            #      "replx", replaced_x_query)
            eval = AstEvaluation(prediction, y_ast_set, y_texts, replaced_x_query,
                                 parse_exception, self.unparser)
            last_x = replaced_x_query
            last_eval_result = eval
            logger.add_evaluation(eval)
            if dump_each:
                if example.id != last_example_id:
                    print("---")
                eval.print_vals(self.unparser)
                last_example_id = example.id

        self.model.set_in_train_mode()

    def data_pair_iterate(self, filter_splits)-> Generator[
        Tuple[XValue, str, AstObjectChoiceSet, ObjectChoiceNode, Set[str], ReplacementSampling],
        None, None
    ]:
        yield from iterate_data_pairs(
            example_store=self.example_store,
            replacers=self.replacer,
            string_parser=self.string_parser,
            str_tokenizer=self.str_tokenizer,
            unparser=self.unparser,
            filter_splits=filter_splits
        )


def make_y_ast_set(
    y_type: AInixType,
    all_y_examples: List[YValue],
    replacement_sample: ReplacementSampling,
    string_parser: StringParser,
    this_x_metadata: StringTokensMetadata,
    unparser: AstUnparser
):
    y_ast_set = AstObjectChoiceSet(y_type, None)
    y_texts = set()
    individual_asts = []
    individual_asts_preferences = []
    for y_example in all_y_examples:
        replaced_y = replacement_sample.replace_x(y_example.y_text)
        if replaced_y not in y_texts:
            parsed_ast = string_parser.create_parse_tree(
                replaced_y, y_type.name)
            individual_asts.append(parsed_ast)
            individual_asts_preferences.append(y_example.y_preference)
            y_ast_set.add(parsed_ast, True, y_example.y_preference, 1.0)
            y_texts.add(replaced_y)
            # handle copies
            # TODO figure how to weight the copy node??
            add_copies_to_ast_set(parsed_ast, y_ast_set, unparser,
                                  this_x_metadata, copy_node_weight=1)
    y_ast_set.freeze()
    teacher_force_path_ast = WeightedRandomChooser(
        individual_asts, individual_asts_preferences).sample()
    return y_ast_set, y_texts, teacher_force_path_ast


def iterate_data_pairs(
    example_store: ExamplesStore,
    replacers: Replacer,
    string_parser: StringParser,
    str_tokenizer: StringTokenizer,
    unparser: AstUnparser,
    filter_splits: Optional[Tuple[DataSplits]]
) -> Generator[
    Tuple[XValue, str, AstObjectChoiceSet, ObjectChoiceNode, Set[str], ReplacementSampling],
    None, None
]:
    """Will yield one epoch of examples as a tuple of the example and the
    Ast set that represents all valid y_values for that example"""
    type_context = example_store.type_context
    all_ex_list = list(example_store.get_all_x_values(filter_splits))
    random.shuffle(all_ex_list)
    for example in all_ex_list:  #self.example_store.get_all_x_values(splits):
        all_y_examples = example_store.get_y_values_for_y_set(example.y_set_id)
        y_type = type_context.get_type_by_name(all_y_examples[0].y_type)
        replacement_sample = replacers.create_replace_sampling(example.x_text)
        this_example_replaced_x = replacement_sample.replace_x(example.x_text)
        this_x_tokens, this_x_metadata = str_tokenizer.tokenize(this_example_replaced_x)
        y_ast_set, y_texts, teacher_force_path_ast = make_y_ast_set(
            y_type, all_y_examples, replacement_sample, string_parser,
            this_x_metadata, unparser
        )
        yield (example, this_example_replaced_x, y_ast_set,
               teacher_force_path_ast, y_texts, replacement_sample)


def flatten_list(lists):
    """https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists"""
    return [item for sublist in lists for item in sublist]


def get_examples(
    split_proportions: SPLIT_PROPORTIONS_TYPE = DEFAULT_SPLITS,
    randomize_seed: bool = False
):
    type_context = TypeContext()
    loader = TypeContextDataLoader(type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(type_context)

    for f in ALL_EXAMPLE_NAMES:
        loader.load_path(f"builtin_types/{f}.ainix.yaml")
    type_context.finalize_data()

    split_seed = None if not randomize_seed else random.randint(1, 1e8)

    index = load_all_examples(type_context, split_proportions, split_seed)
    #index = load_tellina_examples(type_context)
    #index = load_all_and_tellina(type_context)

    #print("num docs", index.get_num_x_values())
    #print("num train", len(list(index.get_all_x_values((DataSplits.TRAIN, )))))

    replacers = get_all_replacers()
    return type_context, index, replacers, loader


# A bunch of code for running the thing which really shouldn't be here.

if __name__ == "__main__":
    argparer = ArgumentParser()
    default_split_train = DEFAULT_SPLITS[0]
    assert default_split_train[1] == DataSplits.TRAIN
    argparer.add_argument("--train_percent", type=float, default=default_split_train[0]*100)
    argparer.add_argument("--randomize_seed", action='store_true')
    argparer.add_argument("--eval_replace_samples", type=int, default=10)
    argparer.add_argument("--quiet_dump", action='store_true')
    argparer.add_argument("--use_pretrain", action='store_true', default=True)
    args = argparer.parse_args()
    train_frac = args.train_percent / 100.0

    #train_frac = 0.05
    #args.randomize_seed = True

    split_proportions = ((train_frac, DataSplits.TRAIN), (1-train_frac, DataSplits.VALIDATION))

    import ainix_kernel.indexing.exampleindex
    from ainix_kernel.indexing import exampleloader
    import datetime

    print("start time", datetime.datetime.now())
    type_context, index, replacers, loader = get_examples(split_proportions, args.randomize_seed)
    from ainix_kernel.models.EncoderDecoder.encdecmodel import \
        get_default_encdec_model

    model = get_default_encdec_model(
        index, standard_size=128, use_retrieval_decoder=False,
        replacer=replacers,
        pretrain_checkpoint=cookie_monster_path() if args.use_pretrain else None,
        learn_on_top_pretrained=args.use_pretrain
    )
    #model = make_rulebased_seacr(index)

    trainer = TypeTranslateCFTrainer(model, index, replacer=replacers, loader=loader)
    train_time = datetime.datetime.now()
    print("train time", train_time)
    epochs = 35
    trainer.train(epochs, eval_every_n_epochs=3,
                  intermitted_save_path="./checkpoints/chkp",
                  dump_each_in_eval=True, intermitted_repl_samples=5)

    print("Lets eval")
    print("-----------")
    print("TRAIN")
    print("-----------")
    logger = EvaluateLogger()
    trainer.evaluate(logger, filter_splits=(DataSplits.TRAIN,), dump_each=not args.quiet_dump,
                     num_replace_samples=10)
    print_ast_eval_log(logger)
    print("-----------")
    print("Validation")
    print("-----------")
    logger = EvaluateLogger()
    trainer.evaluate(logger, dump_each=not args.quiet_dump,
                     num_replace_samples=args.eval_replace_samples)
    print_ast_eval_log(logger)
    print("serialize model")
    serialize(model, loader, "saved_model.pt", logger, trained_epochs=epochs)
    print("done.")
    print("done time", datetime.datetime.now())
    print(datetime.datetime.now() - train_time)
