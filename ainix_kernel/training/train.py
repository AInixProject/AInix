from indexing.examplestore import ExamplesStore
from models.model_types import StringTypeTranslateCF
from parseast import MultitypeStringParser
from training.evaluate import AstEvaluation, EvaluateLogger, print_ast_eval_log


class TypeTranslateCFTrainer:
    def __init__(self, model: StringTypeTranslateCF, example_store: ExamplesStore):
        self.model = model
        self.example_store = example_store
        self.type_context = example_store.type_context
        self.string_parser = MultitypeStringParser(self.type_context)

    def _train_one_epoch(self, which_epoch_on: int):
        for example in self.example_store.get_all_examples():
            self.model

    def train(self, epochs: int):
        for epoch in range(epochs):
            self._train_one_epoch(epoch)

    def evaluate(self, logger: EvaluateLogger):
        for example in self.example_store.get_all_examples():
            # TODO (DNGros): cache / create an all types string parser class
            prediction = self.model.predict(example.xquery, example.ytype)
            expected = self.string_parser.create_parse_tree(
                example.ytext, example.ytype, example.weight)
            logger.add_evaluation(AstEvaluation(prediction, expected))


if __name__ == "__main__":
    from ainix_common.parsing import loader
    from ainix_common.parsing.typecontext import TypeContext
    import indexing.exampleindex
    from indexing import index, exampleloader

    type_context = TypeContext()
    loader.load_path("../../builtin_types/numbers.ainix.yaml", type_context)
    loader.load_path("../../builtin_types/generic_parsers.ainix.yaml", type_context)
    type_context.fill_default_parsers()

    index = indexing.exampleindex.ExamplesIndex(type_context)
    exampleloader.load_path("../../builtin_types/numbers_examples.ainix.yaml", index)

    from models.SeaCR.seacr import SeaCRModel
    model = SeaCRModel(index)
    trainer = TypeTranslateCFTrainer(model, index)

    logger = EvaluateLogger()
    trainer.evaluate(logger)
    print_ast_eval_log(logger)
