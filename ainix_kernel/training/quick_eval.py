"""Toy code for rerunning an eval and playing with a saved model"""
from ainix_kernel.indexing.examplestore import DataSplits
from ainix_kernel.models.EncoderDecoder.encdecmodel import EncDecModel
from ainix_kernel.training.augmenting.replacers import get_all_replacers
from ainix_kernel.training.evaluate import EvaluateLogger, print_ast_eval_log
from ainix_kernel.training.train_contexts import load_all_examples
from ainix_kernel.training.trainer import TypeTranslateCFTrainer
from ainix_kernel.util.serialization import restore

if __name__ == "__main__":
    tc, model, examples = restore("saved_model.pt")
    if examples is None:
        examples = load_all_examples(tc)
    replacer = get_all_replacers()
    logger = EvaluateLogger()
    trainer = TypeTranslateCFTrainer(model, examples, 1, replacer)
    # NOTE: currently when a model is restored it loads loads examples into
    # the latent store. This means if you filter by validation, it isn't really
    # unsean examples. Just examples we didn't see during weight training.
    trainer.evaluate(logger, filter_splits=None, dump_each=True)
    print_ast_eval_log(logger)
