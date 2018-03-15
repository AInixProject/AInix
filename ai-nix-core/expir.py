import torch
from torch import Tensor, nn, optim
from torch.autograd import Variable
import torchtext
import torch.nn.functional as F
import pdb 
import itertools
from ignite.engine import Events
from ignite.trainer import create_supervised_trainer
from ignite.evaluator import Evaluator
from ignite.metrics import CategoricalAccuracy, NegativeLogLikelihood
import math
from ignite.exceptions import NotComputableError


#train_iter, test_iter = torchtext.datasets.IMDB.iters(batch_size=4)
#
#batch = next(iter(train_iter))
#print(batch.text)
#print(batch.label)

# set up fields
TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = torchtext.data.Field(sequential=False)


# make splits for data
train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train, max_size=1000)
LABEL.build_vocab(train)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))
print('len(test)', len(test))

# make iterator for splits
batch_size = 1000
train_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, test), batch_size=batch_size, device=-1)

# print batch information
#batch = next(iter(train_iter))
#print(batch.text)
#print(batch.label)
textVocabLen = len(TEXT.vocab.itos)
labelVocabLen = len(LABEL.vocab.itos)
print("labal vocab", LABEL.vocab.itos)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        embedding_dim = 50
        self.embed = nn.Embedding(textVocabLen, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, labelVocabLen)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(1)
        x = self.fc1(x)
        return F.log_softmax(x)

#optimizer = ...
#model = ...
#criterion = ...
losses = []
criterion = nn.NLLLoss()
model = Net()
optimizer = optim.Adam(model.parameters())
def training_update_function(batch):
    model.train()
    optimizer.zero_grad()
    prediction = model(batch.text[0])
    loss = criterion(prediction, batch.label)
    loss.backward()
    optimizer.step()
    return loss.data[0]

def evaluator_function(batch):
    model.eval()
    prediction = model(batch.text[0])
    return prediction.data.cpu(), batch.label.data.cpu()

from ignite.trainer import Trainer

numOfBatchesPerEpoch = math.ceil(len(train)/batch_size)
numOfBatchesPerEpochVAL = math.ceil(len(test)/batch_size)
val_epoch_iter = itertools.islice(test_iter, numOfBatchesPerEpochVAL)

trainer = Trainer(training_update_function)
evaluator = Evaluator(evaluator_function)
acc = CategoricalAccuracy()
nll = NegativeLogLikelihood()
acc.attach(evaluator, 'accuracy')
nll.attach(evaluator, 'nll')
#
log_interval = 1
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer, state):
    iter = (state.iteration - 1) % numOfBatchesPerEpoch + 1
    if iter % log_interval == 0:
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(state.epoch, iter, numOfBatchesPerEpoch, state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer, state):
    test_iter.repeat = False
    evaluator.run(test_iter)
    avg_accuracy = acc.compute()
    avg_nll = nll.compute()
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(state.epoch, avg_accuracy, avg_nll))
train_iter.repeat = False        
trainer.run(train_iter, max_epochs=20)
