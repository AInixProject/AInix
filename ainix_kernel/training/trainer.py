import random
import torch
from typing import Tuple, Generator

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
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples
from ainix_kernel.util.serialization import serialize


class TypeTranslateCFTrainer:
    def __init__(
        self,
        model: StringTypeTranslateCF,
        example_store: ExamplesStore,
        batch_size: int = 1,
        replacer: Replacer = None
    ):
        self.model = model
        self.example_store = example_store
        self.type_context = example_store.type_context
        self.string_parser = StringParser(self.type_context)
        self.unparser = AstUnparser(self.type_context)
        self.batch_size = batch_size
        self.str_tokenizer = self.model.get_string_tokenizer()
        self.replacer = replacer
        if self.replacer is None:
            self.replacer = Replacer([])

    def _train_one_epoch(self, which_epoch_on: int):
        single_examples_iter = self.data_pair_iterate((DataSplits.TRAIN,))
        batches_iter = more_itertools.chunked(single_examples_iter, self.batch_size)
        loss = torch.tensor(0.0)
        for batch in batches_iter:
            batch_as_query = [(replaced_x, y_ast_set, this_example_ast, example.example_id) for
                              example, replaced_x, y_ast_set, this_example_ast in batch]
            loss += self.model.train_batch(batch_as_query)
        self.model.end_train_epoch()
        return loss

    def train(self, epochs: int):
        self.model.start_train_session()
        for epoch in range(epochs):
            print(f"Start epoch {epoch}")
            loss = self._train_one_epoch(epoch)
            print(f"Epoch {epoch} complete. Loss {loss}")
            if hasattr(self.model, "plz_train_this_latent_store_thanks"):
                # TODO wasdfahwerdfgv I should sleep
                # (yeah, even with sleep to lazy to fix this crappy interface. It works...)
                latent_store = self.model.plz_train_this_latent_store_thanks()
                if latent_store:
                    print("updateding the latent store 🦔")
                    update_latent_store_from_examples(self.model, latent_store, self.example_store,
                                                      self.replacer, self.string_parser,
                                                      (DataSplits.TRAIN,), self.unparser,
                                                      self.str_tokenizer)
            if epoch + 1 != epochs and epoch % 5 == 0:
                print("Pausing to do an eval")
                logger = EvaluateLogger()
                self.evaluate(logger)
                print_ast_eval_log(logger)

        self.model.end_train_session()

    def evaluate(
        self,
        logger: EvaluateLogger,
        filter_splits: Tuple[DataSplits] = (DataSplits.VALIDATION,)
    ):
        self.model.set_in_eval_mode()
        for example, replaced_x_query, y_ast_set, this_example_ast \
                in self.data_pair_iterate(filter_splits):
            try:
                prediction = self.model.predict(replaced_x_query, example.ytype, True)
            except ModelCantPredictException:
                prediction = None
            except ModelSafePredictError:
                prediction = None
            logger.add_evaluation(AstEvaluation(prediction, y_ast_set))
        self.model.set_in_train_mode()

    def data_pair_iterate(
        self,
        filter_splits: Tuple[DataSplits]
    ) -> Generator[Tuple[Example, str, AstObjectChoiceSet, ObjectChoiceNode], None, None]:
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
            for y_example in all_y_examples:
                replaced_y = replacement_sample.replace_x(y_example.ytext)
                parsed_ast = self.string_parser.create_parse_tree(
                    replaced_y, y_type.name)
                if y_example.ytext == example.ytext:
                    ast_for_this_example = parsed_ast
                y_ast_set.add(parsed_ast, True, y_example.weight, 1.0)
            # handle copies
            this_example_replaced_x = replacement_sample.replace_x(example.xquery)
            tokens, metadata = self.str_tokenizer.tokenize(this_example_replaced_x)
            add_copies_to_ast_set(parsed_ast, y_ast_set, self.unparser,
                                  metadata, example.weight)
            y_ast_set.freeze()
            yield (example, this_example_replaced_x, y_ast_set, ast_for_this_example)


# A bunch of code for running the thing which really shouldn't be here.


if __name__ == "__main__":
    from ainix_common.parsing.loader import TypeContextDataLoader
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

    print("num docs", index.backend.index.doc_count())

    from ainix_kernel.models.EncoderDecoder.encdecmodel import \
        get_default_encdec_model

    replacers = get_all_replacers()

    #model = get_default_encdec_model(index, standard_size=64)
    model = get_default_encdec_model(
        index, standard_size=64, replacer=replacers, use_retrieval_decoder=True)
    #model = make_rulebased_seacr(index)

    trainer = TypeTranslateCFTrainer(model, index, replacer=replacers)
    train_time = datetime.datetime.now()
    print("train time", train_time)
    trainer.train(40)

    print("Lets eval")
    logger = EvaluateLogger()
    trainer.evaluate(logger)
    print_ast_eval_log(logger)
    print("serialize model")
    serialize(model, loader, "saved_model.pt")
    print("done.")
    print("done time", datetime.datetime.now())
    print(datetime.datetime.now() - train_time)
