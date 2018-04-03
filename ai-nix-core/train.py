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
from ignite.engines.engine import Events, Engine
from ignite.metrics import CategoricalAccuracy, Loss, Metric
from ignite.exceptions import NotComputableError
import program_description
from cmd_field import CommandField
from run_context import RunContext
import math
import itertools
import torch.nn.functional as F
import data

LOG_INTERVAL = 1

def build_dataset(data, descs, use_cuda):
    inputs, outputs = zip(*data)
    NL_field = torchtext.data.Field(lower=True, include_lengths=True,
        batch_first=True, init_token = None, eos_token = None,
        tensor_type = torch.cuda.LongTensor if use_cuda else torch.LongTensor)
    Command_field = CommandField(descs)

    fields = [('nl', NL_field), ('command', Command_field)]
    examples = [] 
    for x, y in zip(inputs, outputs):
        examples.append(torchtext.data.Example.fromlist([x, y], fields))

    dataset = torchtext.data.Dataset(examples, fields) 
    train, val = dataset.split()

    NL_field.build_vocab(train, max_size=1000)

    return (train, val), fields

def run_train(meta_model, train, val, run_context, test = None):
    batch_size = 1
    train_iter = torchtext.data.iterator.BucketIterator(train,
        batch_size = batch_size, train = True, repeat = False,
        shuffle = True, sort_key=lambda x: x.nl,
        device = None if run_context.use_cuda else -1)
    val_iter = torchtext.data.iterator.BucketIterator(val,
        batch_size = batch_size, train = True, repeat = False,
        shuffle = True, sort_key=lambda x: x.nl,
        device = None if run_context.use_cuda else -1)

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

    trainer = Engine(meta_model.train_step)
    evaluator = Engine(meta_model.eval_step)
    #nll = Loss(F.nll_loss)
    #nll.attach(evaluator, 'nll')
    #acc = CategoricalAccuracy()
    #acc.attach(evaluator, 'accuracy')
    class BashMetric(Metric):
        def reset(self):
            self._num_examples = 0
            self._num_first_commands_right = 0
            self._num_args_seen = 0
            self._num_args_seen_when_right = 0
            self._num_args_pres_correct = 0
            self._num_exact_match = 0

        def update(self, output):
            y_pred, y = output
            self._num_examples += len(y_pred)
            for p, gt in zip(y_pred, y):
                pFirstCmd, gtFirstCmd = p[0], gt[0]
                gotFirstCommand = pFirstCmd.program_desc.name == gtFirstCmd.program_desc.name
                self._num_args_seen += len(gtFirstCmd.arguments)
                if gotFirstCommand:
                    self._num_first_commands_right += 1
                    self._num_args_seen_when_right += len(gtFirstCmd.arguments)
                    fullMatch = True
                    for pArg, gtArg in zip(pFirstCmd.arguments, gtFirstCmd.arguments):
                        if pArg.present == gtArg.present:
                            self._num_args_pres_correct += 1
                        else:
                            fullMatch = False
                    if fullMatch:
                        self._num_exact_match += 1


        def first_cmd_acc(self):
            return self._num_first_commands_right / self._num_examples

        def arg_acc(self):
            return self._num_args_pres_correct / self._num_args_seen

        def exact_match_acc(self):
            return self._num_exact_match / self._num_examples

        def compute(self):
            pass

    bashmetric = BashMetric()
    bashmetric.attach(evaluator, 'bashmetric')



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
        evaluator.run(val_iter)
        #avg_accuracy = acc.compute()
        #avg_nll = nll.compute()
        #print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        #      .format(state.epoch, avg_accuracy, avg_nll))
        print("Validation Results - Epoch: {} FirstCmdAcc: {:.2f} ArgAcc: {:.2f} ExactAcc: {:.2f}"
              .format(state.epoch, bashmetric.first_cmd_acc(), bashmetric.arg_acc(), bashmetric.exact_match_acc()))

    train_iter.repeat = False        
    trainer.run(train_iter, max_epochs=100)

if __name__ == "__main__":
    use_cuda = False #torch.cuda.is_available()
    (train, val), fields = build_dataset(data.all_data, data.all_descs, use_cuda)
    (_, nl_field), (_, cmd_field) = fields 

    STD_WORD_SIZE = 30
    context = RunContext(STD_WORD_SIZE, nl_field, cmd_field, data.all_descs, use_cuda, debug = True)
    meta_model = SimpleCmd(context)

    run_train(meta_model, train, val, context)
