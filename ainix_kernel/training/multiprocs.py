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
import datetime


def example_store_fac(
    type_files: List[str],
    example_files: List[str]
):
    def fac():
        type_context = TypeContext()
        for tf in type_files:
            loader.load_path(tf, type_context)
        type_context.finalize_data()
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
               force_single_thread, eval_thresh, total_proc_count,
               working_count: torch.multiprocessing.Value,
               continue_event, next_cont_event, total_proc_time_accum,
               all_done_event):
    if force_single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads = 1
    print(f"start {pid} actual pid {os.getpid()}")
    #print(f"example count {index.get_doc_count()}")
    #for i in range(10):
    #    print(f"feelin racey? {len(list(index.get_all_examples()))}")
    trainer = TypeTranslateCFTrainer(model, index, batch_size)
    if eval_thresh is None:
        trainer.train(epochs)
    else:
        for e in range(epochs):
            start_time = datetime.datetime.now()
            trainer.train(1)
            should_wait = True
            with working_count.get_lock():
                working_count.value -= 1
                if working_count.value == 0:
                    working_count.value = total_proc_count
                    # everyone done need to measure
                    should_wait = False
                    time_diff = (datetime.datetime.now() - start_time).total_seconds()
                    #print("time diff", time_diff)
                    total_proc_time_accum.value += time_diff
                    #print("total val", total_proc_time_accum.value)
                    logger = EvaluateLogger()
                    trainer.evaluate(logger)
                    acc = logger.stats['ExactMatch'].true_frac
                    print(f"Curr acc {acc}")
                    if acc >= eval_thresh:
                        all_done_event.set()
                    continue_event.set()
                    # swap to new event.
                    # still could have a race, but unlikely
                    continue_event, next_cont_event = next_cont_event, continue_event
                    continue_event.clear()
            if should_wait:
                continue_event.wait()
                continue_event, next_cont_event = next_cont_event, continue_event
            if all_done_event.is_set():
                break


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
        batch_size,
        converge_check_val = None
    ):
        self.model.set_shared_memory()
        all_procs = []
        counter = torch.multiprocessing.Value('i')
        counter.value = workers_count
        val_accum = torch.multiprocessing.Value('d')
        val_accum.value = 0
        event = torch.multiprocessing.Event()
        event.clear()
        other_event = torch.multiprocessing.Event()
        event.clear()
        all_done_event = torch.multiprocessing.Event()
        all_done_event.clear()
        for procid in range(workers_count):
            p = torch.multiprocessing.Process(target=train_func,
                args=(procid,
                      self.model,
                      index,
                      batch_size,
                      epochs_per_worker,
                      self.force_single_thread,
                      converge_check_val,
                      workers_count,
                      counter,
                      event,
                      other_event,
                      val_accum,
                      all_done_event
                      ))
            p.start()
            all_procs.append(p)

        for p in all_procs:
            p.join()

        print("done!")
        return val_accum.value


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
