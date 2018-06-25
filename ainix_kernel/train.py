from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import yaml
import torchtext
import torch
from torch import Tensor, nn, optim
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pudb
from ainix_kernel.model import SimpleCmd
from ignite.engine.engine import Events, Engine
from ignite.metrics import CategoricalAccuracy, Loss, Metric
from ignite.exceptions import NotComputableError
import ainix_kernel.program_description
from ainix_kernel.custom_fields import CommandField, NLField
from ainix_kernel.run_context import RunContext
import math
import itertools
import torch.nn.functional as F
import ainix_kernel.data as sampledata
from ainix_kernel.bashmetrics import BashMetric
from ainix_kernel import constants
import random
from ainix_kernel import tokenizers
import copy, pickle

LOG_INTERVAL = 50

def build_dataset(train, val, descs, use_cuda, test = None):
    datasplits = [x for x in (train, val, test) if x is not None]
    NL_field = NLField(lower=False, include_lengths=True,
        batch_first=True, 
        init_token = constants.SOS, eos_token = constants.EOS, unk_token = constants.UNK,
        tensor_type = torch.cuda.LongTensor if use_cuda else torch.LongTensor,
        tokenize = tokenizers.nonascii_tokenizer)
    Command_field = CommandField(descs, use_cuda)

    fields = [('nl', NL_field), ('command', Command_field)]
    examples_per_split = [[] for l in datasplits]
    datasets = []
    for datasplit, splitexamples in zip(datasplits,examples_per_split):
        inputs, outputs = zip(*datasplit)
        for x, y in zip(inputs, outputs):
            splitexamples.append(torchtext.data.Example.fromlist([x, y], fields))
        datasets.append(torchtext.data.Dataset(splitexamples, fields))

    NL_field.build_vocab(datasets[0], max_size=1000, min_freq = 1)

    return tuple(datasets), fields

def eval_model(meta_model, val_iter, metrics):
    evaluator = Engine(meta_model.eval_step)
    bashmetric = BashMetric()
    for metric, metric_name in metrics:
        metric.attach(evaluator, metric_name)

    evaluator.run(val_iter)


def run_train(meta_model, train_iter, val_iter, run_context, test = None, num_epochs = 50):
    batch = next(iter(train_iter))
    print(batch.nl)
    print(batch.command)
    #textVocabLen = len(NL_field.vocab.itos)
    #print(textVocabLen)
    #labelVocabLen = len(LABEL.vocab.itos)
    #print("nl vocab", NL_field.vocab.itos)
    numOfBatchesPerEpoch = len(train_iter)
    numOfBatchesPerEpochVAL = len(val_iter)

    trainer = Engine(meta_model.train_step)

    if not run_context.quiet_mode:
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            state = engine.state
            iter = (state.iteration - 1) % numOfBatchesPerEpoch + 1
            if iter % LOG_INTERVAL == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(
                    state.epoch, iter, numOfBatchesPerEpoch, state.output))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            state = engine.state
            bashmetric = BashMetric()
            eval_model(meta_model, val_iter, [(bashmetric, 'bashmetric')])
            print("num val examples seen", bashmetric._num_examples)
            print("num values seen", bashmetric._num_values_seen)
            print("""Validation Results - Epoch: {} 
                FirstCmdAcc: {:.2f} ArgAcc: {:.2f} ValExactAcc: {:.2f} ExactAcc: {:.2f}"""
                .format(state.epoch, bashmetric.first_cmd_acc(), bashmetric.arg_acc(),
                    bashmetric.arg_val_exact_acc(), bashmetric.exact_match_acc()))

    train_iter.repeat = False        
    return trainer.run(train_iter, max_epochs=num_epochs)

def run_with_specific_split(train, val, descs, use_cuda, quiet_mode = False, num_epochs = 50):
    (train, val), fields = build_dataset(train, val, descs, use_cuda)
    (_, nl_field), (_, cmd_field) = fields 

    STD_WORD_SIZE = 80
    batch_size = 1
    context = RunContext(STD_WORD_SIZE, nl_field, cmd_field, descs, use_cuda,
            batch_size = batch_size, debug = True, quiet_mode = quiet_mode)

    train_iter = torchtext.data.iterator.BucketIterator(train,
        batch_size = batch_size, train = True, repeat = False,
        shuffle = True, sort_key=lambda x: x.nl,
        device = None if context.use_cuda else -1)
    val_iter = torchtext.data.iterator.BucketIterator(val,
        batch_size = batch_size, train = True, repeat = False,
        shuffle = True, sort_key=lambda x: x.nl,
        device = None if context.use_cuda else -1)

    meta_model = SimpleCmd(context)

    final_state = run_train(meta_model, train_iter, val_iter, context, num_epochs = num_epochs)
    # For now just return everything you could care about in a disorganized messy tupple
    return (meta_model, final_state, train_iter, val_iter)

def run_with_data_list(data, descs, use_cuda, quiet_mode = False, num_epochs = 50, trainsplit = .7):
    """Runs training just based off a single list of (x, y) tupples.
    Will split the data for you"""
    random.shuffle(data)  
    train = data[:int(len(data)*trainsplit)]
    val = data[int(len(data)*trainsplit):]
    return run_with_specific_split(train, val, descs, use_cuda, quiet_mode, num_epochs)

def serialize(meta_model):
    #model_cp = copy.deepcopy(meta_model)
    with open(r"savedtest.pkl", "wb") as output_file:
        pickle.dump(meta_model, output_file)    

if __name__ == "__main__":
    use_cuda = False #torch.cuda.is_available()
    #run_with_data_list(sampledata.all_data, sampledata.all_descs, use_cuda)
    num_train_duplicates = 5
    train, val = sampledata.get_all_data_replaced(num_train_duplicates,2)
    # Uncomment this to use a predefined dataset split. exportforother.py will create this.
    #train, val = sampledata.get_all_data_from_files(
    #        "./splits/src-train.txt", "./splits/trg-train.txt",
    #        "./splits/src-val.txt", "./splits/trg-val.txt")
    # Uncomment this to to use a random split
    meta_model, final_state, train_iter, val_iter = \
        run_with_specific_split(train, val, sampledata.all_descs, use_cuda,
            quiet_mode = False, num_epochs=5//num_train_duplicates)
    serialize(meta_model)

