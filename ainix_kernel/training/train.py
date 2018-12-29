import random
from typing import Tuple, Generator

from ainix_common.parsing.copy_tools import add_copies_to_ast_set
from ainix_kernel.indexing.examplestore import ExamplesStore, DataSplits, Example
from ainix_kernel.models.model_types import StringTypeTranslateCF, ModelCantPredictException, \
    ModelSafePredictError
from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_kernel.training.evaluate import AstEvaluation, EvaluateLogger, print_ast_eval_log
import more_itertools


class TypeTranslateCFTrainer:
    def __init__(
        self,
        model: StringTypeTranslateCF,
        example_store: ExamplesStore,
        batch_size: int = 1
    ):
        self.model = model
        self.example_store = example_store
        self.type_context = example_store.type_context
        self.string_parser = StringParser(self.type_context)
        self.unparser = AstUnparser(self.type_context)
        self.batch_size = batch_size
        self.str_tokenizer = self.model.get_string_tokenizer()

    def _train_one_epoch(self, which_epoch_on: int):
        single_examples_iter = self.data_pair_iterate((DataSplits.TRAIN,))
        batches_iter = more_itertools.chunked(single_examples_iter, self.batch_size)
        for batch in batches_iter:
            batch_as_query = [(example.xquery, y_ast_set, this_example_ast) for
                              example, y_ast_set, this_example_ast in batch]
            self.model.train_batch(batch_as_query)
        self.model.end_train_epoch()

    def train(self, epochs: int):
        self.model.start_train_session()
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            self._train_one_epoch(epoch)
        self.model.end_train_session()

    def evaluate(
        self,
        logger: EvaluateLogger,
        splits: Tuple[DataSplits] = None
    ):
        self.model.end_train_session()
        for example, y_ast_set, this_example_ast in self.data_pair_iterate(splits):
            try:
                prediction = self.model.predict(example.xquery, example.ytype, True)
            except ModelCantPredictException:
                prediction = None
            except ModelSafePredictError:
                prediction = None
            logger.add_evaluation(AstEvaluation(prediction, y_ast_set))

    def data_pair_iterate(
        self,
        splits: Tuple[DataSplits]
    ) -> Generator[Tuple[Example, AstObjectChoiceSet, ObjectChoiceNode], None, None]:
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
            for y_example in all_y_examples:
                parsed_ast = self.string_parser.create_parse_tree(
                    y_example.ytext, y_type.name)
                if y_example.ytext == example.ytext:
                    ast_for_this_example = parsed_ast
                y_ast_set.add(parsed_ast, True, y_example.weight, 1.0)
            tokens, metadata = self.str_tokenizer.tokenize(example.xquery)
            add_copies_to_ast_set(parsed_ast, y_ast_set, self.unparser,
                                  metadata, example.weight)
            y_ast_set.freeze()
            yield (example, y_ast_set, ast_for_this_example)


if __name__ == "__main__":
    from ainix_common.parsing import loader
    from ainix_common.parsing.typecontext import TypeContext
    import ainix_kernel.indexing.exampleindex
    from ainix_kernel.indexing import exampleloader
    import datetime

    print("start time", datetime.datetime.now())
    type_context = TypeContext()
    loader.load_path("builtin_types/numbers.ainix.yaml", type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml", type_context, up_search_limit=4)
    loader.load_path("builtin_types/command.ainix.yaml", type_context, up_search_limit=4)
    loader.load_path("builtin_types/pwd.ainix.yaml", type_context, up_search_limit=4)
    loader.load_path("builtin_types/ls.ainix.yaml", type_context, up_search_limit=4)
    type_context.fill_default_parsers()

    index = ainix_kernel.indexing.exampleindex.ExamplesIndex(type_context)
    exampleloader.load_path("../../builtin_types/numbers_examples.ainix.yaml", index)
    exampleloader.load_path("../../builtin_types/pwd_examples.ainix.yaml", index)
    exampleloader.load_path("../../builtin_types/ls_examples.ainix.yaml", index)
    print("num docs", index.backend.index.doc_count())

    from ainix_kernel.models.SeaCR.seacr import make_default_seacr, make_rulebased_seacr
    from ainix_kernel.models.EncoderDecoder.encdecmodel import get_default_encdec_model
    model = get_default_encdec_model(index, standard_size=64)
    #model = make_rulebased_seacr(index)
    trainer = TypeTranslateCFTrainer(model, index)
    train_time = datetime.datetime.now()
    print("train time", train_time)
    trainer.train(30)

    print("Lets eval")
    logger = EvaluateLogger()
    trainer.evaluate(logger)
    print_ast_eval_log(logger)
    print("done.")
    print("done time", datetime.datetime.now())
    print(datetime.datetime.now() - train_time)
