import random
import torch
from typing import Tuple, Generator, List, Set, Optional

from ainix_common.parsing.copy_tools import add_copies_to_ast_set
from ainix_kernel.indexing.examplestore import ExamplesStore, DataSplits, Example
from ainix_kernel.models.model_types import StringTypeTranslateCF, ModelCantPredictException, \
    ModelSafePredictError
from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_kernel.training.augmenting.replacers import Replacer, get_all_replacers
from ainix_kernel.training.evaluate import AstEvaluation, EvaluateLogger, print_ast_eval_log
import more_itertools
from ainix_kernel.specialtypes import allspecials
from ainix_kernel.training.model_specific_training import update_latent_store_from_examples
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples, \
    load_tellia_examples
from ainix_kernel.util.serialization import serialize
from tqdm import tqdm
from ainix_common.parsing.loader import TypeContextDataLoader


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
        train_count = sum(1 for _ in self.example_store.get_all_examples(train_splits))
        single_examples_iter = self.data_pair_iterate(train_splits)
        batches_iter = more_itertools.chunked(single_examples_iter, self.batch_size)
        loss = torch.tensor(0.0)
        examples_seen = 0
        with tqdm(total=train_count, unit='Examples', miniters=train_count/5) as pbar:
            for batch in batches_iter:
                batch_as_query = [(replaced_x, y_ast_set, this_example_ast, example.example_id) for
                                  example, replaced_x, y_ast_set, this_example_ast, ytxts in batch]
                loss += float(self.model.train_batch(batch_as_query))
                examples_seen += len(batch)
                pbar.update(len(batch))
        self.model.end_train_epoch()
        return loss / examples_seen

    def train(self, epochs: int, eval_every_n_epochs: int = None, intermitted_save_path = None):
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
            if eval_every_n_epochs and epoch + 1 != epochs and epoch % eval_every_n_epochs == 0:
                print("Pausing to do an eval")
                logger = EvaluateLogger()
                self.evaluate(logger, dump_each=True)
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
        dups.sort(key=lambda d: d[0].example_id)

        last_x = None
        last_eval_result = None
        last_example_id = None
        for data in dups:
            example, replaced_x_query, y_ast_set, this_example_ast, y_texts = data
            if last_x == replaced_x_query:
                logger.add_evaluation(last_eval_result)
                continue
            parse_exception = None
            try:
                prediction, metad = self.model.predict(replaced_x_query, example.ytype, True)
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
                if example.example_id != last_example_id:
                    print("---")
                eval.print_vals(self.unparser)
                last_example_id = example.example_id

        self.model.set_in_train_mode()

    def data_pair_iterate(
        self,
        filter_splits: Tuple[DataSplits]
    ) -> Generator[
        Tuple[Example, str, AstObjectChoiceSet, ObjectChoiceNode, Set[str]], None, None
    ]:
        """Will yield one epoch of examples as a tuple of the example and the
        Ast set that represents all valid y_values for that example"""
        # Temporary hack for shuffling
        # TODO (DNGros): Make suffle up to a buffer or something
        all_ex_list = list(self.example_store.get_all_examples(filter_splits))
        random.shuffle(all_ex_list)
        #
        for example in all_ex_list:  #self.example_store.get_all_examples(splits):
            all_y_examples = self.example_store.get_examples_from_y_set(example.y_set_id)
            y_type = self.type_context.get_type_by_name(example.ytype)
            y_ast_set = AstObjectChoiceSet(y_type, None)
            ast_for_this_example = None
            parsed_ast = None
            replacement_sample = self.replacer.create_replace_sampling(example.xquery)
            y_texts = set()
            for y_example in all_y_examples:
                replaced_y = replacement_sample.replace_x(y_example.ytext)
                if replaced_y not in y_texts:
                    parsed_ast = self.string_parser.create_parse_tree(
                        replaced_y, y_type.name)
                    if y_example.ytext == example.ytext:
                        ast_for_this_example = parsed_ast
                    y_ast_set.add(parsed_ast, True, y_example.weight, 1.0)
                    y_texts.add(replaced_y)
            # handle copies
            this_example_replaced_x = replacement_sample.replace_x(example.xquery)
            tokens, metadata = self.str_tokenizer.tokenize(this_example_replaced_x)
            add_copies_to_ast_set(parsed_ast, y_ast_set, self.unparser,
                                  metadata, example.weight)
            y_ast_set.freeze()
            yield (example, this_example_replaced_x, y_ast_set, ast_for_this_example, y_texts)


# A bunch of code for running the thing which really shouldn't be here.


if __name__ == "__main__":
    from ainix_common.parsing.typecontext import TypeContext
    import ainix_kernel.indexing.exampleindex
    from ainix_kernel.indexing import exampleloader
    import datetime

    print("start time", datetime.datetime.now())
    type_context = TypeContext()
    loader = TypeContextDataLoader(type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(type_context)

    for f in ALL_EXAMPLE_NAMES:
        loader.load_path(f"builtin_types/{f}.ainix.yaml")
    type_context.finalize_data()

    #exampleloader.load_path(f"../../builtin_types/why_not_work_examples.ainix.yaml", index)
    index = load_all_examples(type_context)
    #index = load_tellia_examples(type_context)

    print("num docs", index.backend.index.doc_count())

    from ainix_kernel.models.EncoderDecoder.encdecmodel import \
        get_default_encdec_model

    replacers = get_all_replacers()

    model = get_default_encdec_model(
        index, standard_size=200, use_retrieval_decoder=False, replacer=replacers,
        pretrain_checkpoint="../../checkpoints/lmchkp_iter152k_200_2rnn_total3.29_ns0.47_lm2.82.pt")

    #t model = get_default_encdec_model(
    #    index, standard_size=64, replacer=replacers, use_retrieval_decoder=True)

    #model = make_rulebased_seacr(index)

    trainer = TypeTranslateCFTrainer(model, index, replacer=replacers, loader=loader)
    train_time = datetime.datetime.now()
    print("train time", train_time)
    epochs = 40
    trainer.train(epochs, eval_every_n_epochs=2, intermitted_save_path="./checkpoints/chkp")

    print("Lets eval")
    print("-----------")
    print("TRAIN")
    print("-----------")
    logger = EvaluateLogger()
    trainer.evaluate(logger, filter_splits=(DataSplits.TRAIN,), dump_each=True)
    print_ast_eval_log(logger)
    print("-----------")
    print("Validation")
    print("-----------")
    logger = EvaluateLogger()
    trainer.evaluate(logger, dump_each=True)
    print_ast_eval_log(logger)
    print("serialize model")
    serialize(model, loader, "saved_model.pt", logger, trained_epochs=epochs)
    print("done.")
    print("done time", datetime.datetime.now())
    print(datetime.datetime.now() - train_time)


def flatten_list(lists):
    """https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists"""
    return [item for sublist in lists for item in sublist]
