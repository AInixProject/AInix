"""Poorly written code for plots for concurrency

TODO (DNGros): Cleanup / remove before publishing
"""
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
import numpy as np
import sys


def get_examples_per_sec(batch_size, examples_sample_size=1e4):
    print("start time", datetime.datetime.now())
    type_context = TypeContext()
    loader.load_path("../../builtin_types/numbers.ainix.yaml", type_context)
    loader.load_path("../../builtin_types/generic_parsers.ainix.yaml", type_context)
    type_context.fill_default_parsers()

    index = ainix_kernel.indexing.exampleindex.ExamplesIndex(type_context)
    exampleloader.load_path("../../builtin_types/numbers_examples.ainix.yaml", index)
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
            examples_per_sec = get_examples_per_sec(batch_size, 500)
            print(f"speed {examples_per_sec}")
            list_data.append({
                "batch_size": batch_size,
                "examples_per_sec": examples_per_sec
            })
    data = pd.DataFrame(list_data)
    plot_data(data, title="Batched Encoder Throughput",
              x="batch_size", y="examples_per_sec",
              file_name="./figures/encoder_batched_speed.png")


if __name__ == "__main__":
    import os
    script_path = os.path.dirname(os.path.abspath(__file__))
    print(script_path)
    fig_batchsize_vs_throughput()
