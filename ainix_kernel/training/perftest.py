"""Poorly written code for plots for concurrency

TODO (DNGros): Cleanup / move / document before publish
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.training import train
from ainix_common.parsing import loader
from ainix_common.parsing.typecontext import TypeContext
import ainix_kernel.indexing.exampleindex
from ainix_kernel.indexing import exampleloader
import datetime
from ainix_kernel.models.EncoderDecoder.encdecmodel import get_default_encdec_model
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import sys

from ainix_kernel.training.evaluate import EvaluateLogger, print_ast_eval_log
from ainix_kernel.training.multiprocs import example_store_fac, MultiprocTrainer, \
    make_default_trainer_fac
from ainix_kernel.training.train import TypeTranslateCFTrainer


def get_index() -> ExamplesStore:
    index_fac = example_store_fac([
        '../../builtin_types/numbers.ainix.yaml',
        '../../builtin_types/generic_parsers.ainix.yaml'
    ], [
        "../../builtin_types/numbers_examples.ainix.yaml"
    ])
    index = index_fac()
    return index


def get_examples_per_sec(batch_size, examples_sample_size=1e4):
    index = get_index()
    num_docs = index.get_doc_count()
    print("num docs", num_docs)

    model = get_default_encdec_model(index, standard_size=64)
    trainer = train.TypeTranslateCFTrainer(model, index, batch_size=batch_size)
    train_time = datetime.datetime.now()
    print("train time", train_time)

    epochs = math.ceil(examples_sample_size / num_docs)
    print("going to perform", epochs, "epochs")
    trainer.train(epochs)
    seconds_taken = (datetime.datetime.now() - train_time).total_seconds()
    examples_done = num_docs * epochs
    examples_per_second = examples_done / seconds_taken
    return examples_per_second


def train_to_threshold_st(batch_size, threshold=0.7, max_epochs=100):
    index = get_index()
    num_docs = index.get_doc_count()
    print("num docs", num_docs)

    model = get_default_encdec_model(index, standard_size=64)
    trainer = train.TypeTranslateCFTrainer(model, index, batch_size=batch_size)
    time_spent = 0
    for epoch in range(max_epochs):
        start_time = datetime.datetime.now()
        trainer.train(1)
        time_spent += (datetime.datetime.now() - start_time).total_seconds()
        logger = EvaluateLogger()
        trainer.evaluate(logger)
        acc = logger.stats['ExactMatch'].true_frac
        print(f"Curr acc {acc}")
        if acc >= threshold:
            break

    return time_spent


def plot_data(data, title=None, file_name=None, x="str_len", y="run_time_ms", hue=None):
    sns.set()
    sns.barplot(x=x, y=y, hue=hue, data=data) #, estimator=min, ci=68)
    plt.title(title)
    if file_name:
        plt.savefig(file_name)
    plt.show()


def fig_batchsize_vs_throughput():
    sizes_to_test = [1, 4, 8, 16, 32]
    list_data = []
    for rerun in range(3):
        for batch_size in sizes_to_test:
            print(f"running with batch size {batch_size}")
            examples_per_sec = get_examples_per_sec(batch_size, 5000)
            print(f"speed {examples_per_sec}")
            list_data.append({
                "batch_size": batch_size,
                "examples_per_sec": examples_per_sec
            })
    data = pd.DataFrame(list_data)
    plot_data(data, title="Batched Encoder Throughput",
              x="batch_size", y="examples_per_sec",
              file_name="./figures/encoder_batched_speed.png")


def fig_batchsize_vs_converg():
    sizes_to_test = [1, 4, 8, 16, 32]
    #sizes_to_test = [1, 4]
    list_data = []
    for rerun in range(1):
        for batch_size in sizes_to_test:
            print(f"running with batch size {batch_size}")
            train_time = train_to_threshold_st(batch_size, threshold=0.75)
            print(f"time {train_time}")
            list_data.append({
                "batch_size": batch_size,
                "train_time": train_time
            })
    data = pd.DataFrame(list_data)
    plot_data(data, title="Batched Encoder Train Time",
              x="batch_size", y="train_time",
              file_name="./figures/encoder_batched_train_time.png")


def get_examples_per_sec_multi(batch_size, procs, examples_sample_size=1e4):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads = 1

    index = get_index()
    print("num docs", index.get_doc_count())
    model = get_default_encdec_model(examples=index, standard_size=64)
    # train
    mptrainer = MultiprocTrainer(
        model,
        make_default_trainer_fac(
            model,
            batch_size
        )
    )
    num_docs = index.get_doc_count()
    epochs = math.ceil(examples_sample_size / num_docs / procs)
    print("going to perform", epochs, "epochs")
    train_time = datetime.datetime.now()
    mptrainer.train(workers_count=procs, epochs_per_worker=epochs,
                    index=index, batch_size=batch_size)
    seconds_taken = (datetime.datetime.now() - train_time).total_seconds()
    examples_done = num_docs * epochs * procs
    examples_per_second = examples_done / seconds_taken
    return examples_per_second


def get_multi_train_time(batch_size, procs):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads = 1

    index = get_index()
    print("num docs", index.get_doc_count())
    model = get_default_encdec_model(examples=index, standard_size=64)
    # train
    mptrainer = MultiprocTrainer(
        model,
        make_default_trainer_fac(
            model,
            batch_size
        )
    )
    num_docs = index.get_doc_count()
    time = mptrainer.train(workers_count=procs, epochs_per_worker=50,
                           index=index, batch_size=batch_size, converge_check_val=0.75)
    return time


def fig_multiproc_vs_throughput():
    sizes_to_test = [1, 4, 8, 12]
    #sizes_to_test = [1]
    list_data = []
    batch_size = 1
    for rerun in range(3):
        for procs in sizes_to_test:
            print(f"running with procs {procs}")
            examples_per_sec = get_examples_per_sec_multi(batch_size, procs, 5000)
            print(f"speed {examples_per_sec}")
            list_data.append({
                "batch_size": batch_size,
                "processes": procs,
                "examples_per_sec": examples_per_sec
            })
    data = pd.DataFrame(list_data)
    plot_data(data, title="Multiprocess Model Training",
              x="processes", y="examples_per_sec",
              file_name="./figures/multi_batched_speed.png")


def fig_multiproc_vs_time():
    sizes_to_test = [1, 4, 8, 12]
    #sizes_to_test = [1]
    list_data = []
    batch_size = 1
    for rerun in range(3):
        for procs in sizes_to_test:
            print(f"running with procs {procs}")
            rt = get_multi_train_time(batch_size, procs)
            print(f"time {rt}")
            list_data.append({
                "batch_size": batch_size,
                "processes": procs,
                "run_time": rt
            })
    data = pd.DataFrame(list_data)
    plot_data(data, title="Multiprocess Model Training Time",
              x="processes", y="run_time",
              file_name="./figures/multi_time.png")


if __name__ == "__main__":
    import os
    script_path = os.path.dirname(os.path.abspath(__file__))
    print(script_path)
    #fig_batchsize_vs_throughput()
    fig_multiproc_vs_throughput()
    #fig_batchsize_vs_converg()
    #print(get_multi_train_time(1, 2))
    #fig_multiproc_vs_time()
