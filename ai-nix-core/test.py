import torch
from torch import Tensor, nn, optim
from torch.autograd import Variable
import torchtext
import torch.nn.functional as F
import pdb
import itertools


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
        embedding_dim = 5
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
    print("step loss", loss)
    loss.backward()
    optimizer.step()
    return loss.data[0]

from ignite.trainer import Trainer

trainer = Trainer(training_update_function)
trainer.run(itertools.islice(train_iter, len(train)//batch_size), max_epochs=5)
