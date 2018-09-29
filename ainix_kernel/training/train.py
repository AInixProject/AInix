from typing import Tuple, Generator

from indexing.examplestore import ExamplesStore, DataSplits, Example
from ainix_kernel.models.model_types import StringTypeTranslateCF, ModelCantPredictException
from ainix_common.parsing.parseast import StringParser, AstObjectChoiceSet, ObjectChoiceNode
from training.evaluate import AstEvaluation, EvaluateLogger, print_ast_eval_log


class TypeTranslateCFTrainer:
    def __init__(self, model: StringTypeTranslateCF, example_store: ExamplesStore):
        self.model = model
        self.example_store = example_store
        self.type_context = example_store.type_context
        self.string_parser = StringParser(self.type_context)

    def _train_one_epoch(self, which_epoch_on: int):
        for example, y_ast_set, this_example_ast in self.data_pair_iterate((DataSplits.TRAIN,)):
            self.model.train(example.xquery, y_ast_set, this_example_ast)

    def train(self, epochs: int):
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            self._train_one_epoch(epoch)

    def evaluate(
        self,
        logger: EvaluateLogger,
        splits: Tuple[DataSplits] = None
    ):
        for example, y_ast_set, this_example_ast in self.data_pair_iterate(splits):
            #this_ex_p = self.string_parser.create_parse_tree(example.ytext, example.ytype)
            #assert y_ast_set.is_node_known_valid(this_ex_p)
            try:
                prediction = self.model.predict(example.xquery, example.ytype, True)
            except ModelCantPredictException:
                prediction = None

            #if prediction == this_ex_p:
            #    print("YAT")
            #    assert y_ast_set.is_node_known_valid(prediction)

            logger.add_evaluation(AstEvaluation(prediction, y_ast_set))

    def data_pair_iterate(
        self,
        splits: Tuple[DataSplits]
    ) -> Generator[Tuple[Example, AstObjectChoiceSet, ObjectChoiceNode], None, None]:
        """Will yield one epoch of examples as a tuple of the example and the
        Ast set that represents all valid y_values for that example"""
        for example in self.example_store.get_all_examples(splits):
            all_y_examples = self.example_store.get_examples_from_y_set(example.y_set_id)
            y_type = self.type_context.get_type_by_name(example.ytype)
            y_ast_set = AstObjectChoiceSet(y_type, None)
            ast_for_this_example = None
            for y_example in all_y_examples:
                parsed_ast = self.string_parser.create_parse_tree(
                    y_example.ytext, y_type.name)
                if y_example.ytext == example.ytext:
                    ast_for_this_example = parsed_ast
                y_ast_set.add(parsed_ast, True, y_example.weight, 1.0)
            y_ast_set.freeze()
            yield (example, y_ast_set, ast_for_this_example)


if __name__ == "__main__":
    from ainix_common.parsing import loader
    from ainix_common.parsing.typecontext import TypeContext
    import indexing.exampleindex
    from indexing import exampleloader

    type_context = TypeContext()
    loader.load_path("../../builtin_types/numbers.ainix.yaml", type_context)
    loader.load_path("../../builtin_types/generic_parsers.ainix.yaml", type_context)
    type_context.fill_default_parsers()

    index = indexing.exampleindex.ExamplesIndex(type_context)
    exampleloader.load_path("../../builtin_types/numbers_examples.ainix.yaml", index)
    print("num docs", index.backend.index.doc_count())

    from models.SeaCR.seacr import SeaCRModel
    model = SeaCRModel(index)
    trainer = TypeTranslateCFTrainer(model, index)

    logger = EvaluateLogger()
    trainer.evaluate(logger)
    print_ast_eval_log(logger)
