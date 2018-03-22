import yaml
import torchtext
import torch
from torch import Tensor, nn, optim
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pudb
from model import SimpleCmd
from ignite.engines.engine import Events
from ignite.engines.trainer import Trainer
from ignite.engines.evaluator import Evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from ignite.exceptions import NotComputableError
import program_description
from cmd_field import CommandField
from run_context import RunContext
import math
import itertools
import torch.nn.functional as F

LOG_INTERVAL = 1

def build_dataset(data, descs):
    inputs, outputs = zip(*data)
    NL_field = torchtext.data.Field(lower=True, include_lengths=True,
        batch_first=True, init_token = None, eos_token = None)
    Command_field = CommandField(descs)

    fields = [('nl', NL_field), ('command', Command_field)]
    examples = []

    for x, y in zip(inputs, outputs):
        examples.append(torchtext.data.Example.fromlist([x, y], fields))

    dataset = torchtext.data.Dataset(examples, fields) 
    train, val = dataset.split()

    NL_field.build_vocab(train, max_size=1000)

    return (train, val), fields

def run_train(meta_model, train, val, test = None):
    batch_size = 1
    train_iter = torchtext.data.iterator.BucketIterator(train,
        batch_size = batch_size, train = True, repeat = False,
        shuffle = True, sort_key=lambda x: x.nl)
    val_iter = torchtext.data.iterator.BucketIterator(val,
        batch_size = batch_size, train = True, repeat = False,
        shuffle = True, sort_key=lambda x: x.nl)

    batch = next(iter(train_iter))
    print(batch.nl)
    print(batch.command)
    #textVocabLen = len(NL_field.vocab.itos)
    #print(textVocabLen)
    #labelVocabLen = len(LABEL.vocab.itos)
    #print("nl vocab", NL_field.vocab.itos)
    numOfBatchesPerEpoch = math.ceil(len(train)/batch_size)
    numOfBatchesPerEpochVAL = math.ceil(len(val)/batch_size)
    val_epoch_iter = itertools.islice(val_iter, numOfBatchesPerEpochVAL)

    trainer = Trainer(meta_model.train_step)
    evaluator = Evaluator(meta_model.eval_step)
    nll = Loss(F.nll_loss)
    nll.attach(evaluator, 'nll')
    acc = CategoricalAccuracy()
    acc.attach(evaluator, 'accuracy')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer, state):
        iter = (state.iteration - 1) % numOfBatchesPerEpoch + 1
        if iter % LOG_INTERVAL == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(
                state.epoch, iter, numOfBatchesPerEpoch, state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer, state):
        evaluator.run(val_iter)
        avg_accuracy = acc.compute()
        avg_nll = nll.compute()
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(state.epoch, avg_accuracy, avg_nll))

    train_iter.repeat = False        
    trainer.run(train_iter, max_epochs=30)

    #acc = CategoricalAccuracy()
    #acc.attach(evaluator, 'accuracy')

    #@trainer.on(Events.EPOCH_COMPLETED)
    #def log_validation_results(trainer, state):
    #    test_iter.repeat = False
    #    evaluator.run(test_iter)
    #    avg_accuracy = acc.compute()
    #    avg_nll = nll.compute()
    #    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #          .format(state.epoch, avg_accuracy, avg_nll))
        

lsDesc = program_description.AIProgramDescription(
    name = "ls"
)
pwdDesc = program_description.AIProgramDescription(
    name = "pwd"
)
if __name__ == "__main__":
    data = [
        ("list all files", "ls"),
        ("list all files here", "ls"),
        ("what file am I in", "pwd"),
        ("list my files", "ls"),
        ("list what is here", "ls"),
        ("list files and dirs here", "ls"),
        ("list", "ls"),
        ("print working directory", "pwd"),
        ("print current dir", "pwd")
    ]
    descs = [lsDesc, pwdDesc]
    (train, val), fields = build_dataset(data, descs)
    (_, nl_field), (_, cmd_field) = fields 

    STD_WORD_SIZE = 10
    use_cuda = torch.cuda.is_available()
    context = RunContext(STD_WORD_SIZE, nl_field, cmd_field, descs, use_cuda)
    meta_model = SimpleCmd(context)

    run_train(meta_model, train, val)
