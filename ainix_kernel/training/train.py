import random
import torch
from typing import Tuple, Generator

from ainix_common.parsing.copy_tools import add_copies_to_ast_set
from ainix_kernel.indexing.examplestore import ExamplesStore, DataSplits, Example
from ainix_kernel.models.EncoderDecoder.latentstore import LatentStore
from ainix_kernel.models.model_types import StringTypeTranslateCF, ModelCantPredictException, \
    ModelSafePredictError
from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_kernel.training.augmenting.replacers import Replacer, ReplacementGroup, Replacement
from ainix_kernel.training.evaluate import AstEvaluation, EvaluateLogger, print_ast_eval_log
import more_itertools
from ainix_kernel.specialtypes import generic_strings, allspecials
from ainix_kernel.util.serialization import serialize
from ainix_kernel.models.EncoderDecoder.encdecmodel import EncDecModel


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
        for batch in batches_iter:
            batch_as_query = [(replaced_x, y_ast_set, this_example_ast, example.example_id) for
                              example, replaced_x, y_ast_set, this_example_ast in batch]
            self.model.train_batch(batch_as_query)
        self.model.end_train_epoch()

    def train(self, epochs: int):
        self.model.start_train_session()
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            self._train_one_epoch(epoch)
            if hasattr(self.model, "plz_train_this_latent_store_thanks"):
                # TODO wasdfahwerdfgv I should sleep
                latent_store = self.model.plz_train_this_latent_store_thanks()
                if latent_store:
                    print("updatedin the thing 🦔🦔")
                    update_latent_store_from_examples(self.model, latent_store, self.example_store,
                                                      self.replacer, self.string_parser,
                                                      (DataSplits.TRAIN,))

        self.model.end_train_session()

    def evaluate(
        self,
        logger: EvaluateLogger,
        splits: Tuple[DataSplits] = None
    ):
        self.model.set_in_eval_mode()
        for example, replaced_x_query, y_ast_set, this_example_ast \
                in self.data_pair_iterate(splits):
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
        splits: Tuple[DataSplits]
    ) -> Generator[Tuple[Example, str, AstObjectChoiceSet, ObjectChoiceNode], None, None]:
        """Will yield one epoch of examples as a tuple of the example and the
        Ast set that represents all valid y_values for that example"""
        # Temporary hack for shuffling
        # TODO (DNGros): Make suffle up to a buffer or something
        all_ex_list = list(self.example_store.get_all_examples(splits))
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


# Latent store update code because I don't know where else to put it

def update_latent_store_from_examples(
    model: EncDecModel,
    latent_store: LatentStore,
    examples: ExamplesStore,
    replacer: Replacer,
    parser: StringParser,
    splits: Tuple[DataSplits]
):
    model.set_in_eval_mode()
    for example in examples.get_all_examples(splits):
        # TODO multi sampling replacers
        x, y = replacer.strings_replace(example.xquery, example.ytext)
        if x != example.xquery or y != example.ytext:
            raise NotImplemented("need to implement copying")
        ast = parser.create_parse_tree(y, example.ytype)
        # TODO: add copies
        latents = model.get_latent_select_states(x, ast)
        nodes = list(ast.depth_first_iter())
        #print("LATENTS", latents)
        for i, l in enumerate(latents):
            dfs_depth = i*2
            n = nodes[dfs_depth].cur_node
            assert isinstance(n, ObjectChoiceNode)
            c = l.detach()
            assert not c.requires_grad
            latent_store.set_latent_for_example(c, n.type_to_choose.ind,
                                                example.example_id, dfs_depth)
    model.set_in_train_mode()


# A bunch of code for running the thing which really shouldn't be here.

def get_all_replacers() -> Replacer:
    filename_repl = ReplacementGroup('FILENAME', Replacement.from_tsv("./data/FILENAME.tsv"))
    dirname_repl = ReplacementGroup('DIRNAME', Replacement.from_tsv("./data/DIRNAME.tsv"))
    eng_word_repl = ReplacementGroup('ENGWORD', Replacement.from_tsv("./data/ENGWORD.tsv"))
    replacer = Replacer([filename_repl, dirname_repl, eng_word_repl])
    return replacer


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

    with_example_files = ("numbers", "pwd", "ls", "cat", "head", "cp", "wc",
                          "mkdir", "echo", "mv", "touch")
    for f in with_example_files:
        loader.load_path(f"builtin_types/{f}.ainix.yaml")
    type_context.finalize_data()

    index = ainix_kernel.indexing.exampleindex.ExamplesIndex(type_context)
    for f in with_example_files:
        exampleloader.load_path(f"../../builtin_types/{f}_examples.ainix.yaml", index)

    print("num docs", index.backend.index.doc_count())

    from ainix_kernel.models.SeaCR.seacr import make_default_seacr, make_rulebased_seacr
    from ainix_kernel.models.EncoderDecoder.encdecmodel import get_default_encdec_model, EncDecModel, \
    EncDecModel, EncDecModel, EncDecModel

    model = get_default_encdec_model(index, standard_size=64)
    #model = make_rulebased_seacr(index)

    trainer = TypeTranslateCFTrainer(model, index, replacer=get_all_replacers())
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
