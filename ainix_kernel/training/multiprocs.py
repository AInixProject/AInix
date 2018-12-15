"""Code for multiproccessing training

TODO (DNGros): This code is a disgusting mess. Clean before pushing.
"""
from typing import List

from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.models.model_types import StringTypeTranslateCF
from ainix_kernel.training import train
import torch
from ainix_kernel.models.EncoderDecoder.encdecmodel import get_default_encdec_model
from ainix_common.parsing import loader
from ainix_kernel.indexing import exampleloader
import ainix_kernel.indexing.exampleindex
import os

from ainix_kernel.training.evaluate import EvaluateLogger, print_ast_eval_log
from ainix_kernel.training.train import TypeTranslateCFTrainer


def example_store_fac(
    type_files: List[str],
    example_files: List[str]
):
    def fac():
        type_context = TypeContext()
        for tf in type_files:
            loader.load_path(tf, type_context)
        type_context.fill_default_parsers()
        index = ainix_kernel.indexing.exampleindex.ExamplesIndex(type_context)
        for ef in example_files:
            exampleloader.load_path(ef, index)
        return index
    return fac

def make_default_trainer_fac(
    model: StringTypeTranslateCF,
    batch_size: int
):
    print("hello there")


def bound_torch_threads(procs):
    pass


def train_func(pid, model: StringTypeTranslateCF, index: ExamplesStore, batch_size, epochs,
               force_single_thread):
    if force_single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads = 1
    print(f"start {pid} actual pid {os.getpid()}")
    #print(f"example count {index.get_doc_count()}")
    #for i in range(10):
    #    print(f"feelin racey? {len(list(index.get_all_examples()))}")
    trainer = TypeTranslateCFTrainer(model, index, batch_size)
    trainer.train(epochs)


class MultiprocTrainer:
    def __init__(
        self,
        model: StringTypeTranslateCF,
        trainer_factory,
        force_single_thread = False
    ):
        self.model = model
        self.trainer_factory = trainer_factory
        self.force_single_thread = False

    def train(
        self,
        workers_count: int,
        epochs_per_worker: int,
        index: ExamplesStore,
        batch_size
    ):
        self.model.set_shared_memory()
        all_procs = []
        for procid in range(workers_count):
            p = torch.multiprocessing.Process(target=train_func,
                args=(procid, self.model, index, batch_size, epochs_per_worker,
                      self.force_single_thread))
            p.start()
            all_procs.append(p)

        for p in all_procs:
            p.join()

        print("done!")


if __name__ == "__main__":
    index_fac = example_store_fac([
        '../../builtin_types/numbers.ainix.yaml',
        '../../builtin_types/generic_parsers.ainix.yaml'
    ], [
        "../../builtin_types/numbers_examples.ainix.yaml"
    ])
    batch_size = 4
    index = index_fac()
    model = get_default_encdec_model(examples=index)
    # Try before
    print("before:")
    trainer = TypeTranslateCFTrainer(model, index, batch_size)
    logger = EvaluateLogger()
    trainer.evaluate(logger)
    print_ast_eval_log(logger)

    # train

    mptrainer = MultiprocTrainer(
        model,
        make_default_trainer_fac(
            model,
            batch_size
        )
    )
    mptrainer.train(5, 10, index, batch_size)

    logger = EvaluateLogger()
    trainer.evaluate(logger)
    print("after:")
    print_ast_eval_log(logger)
