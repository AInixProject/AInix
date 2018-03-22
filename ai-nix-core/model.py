import torch
from torch import Tensor, nn, optim
from torch.autograd import Variable
import torchtext
import torch.nn.functional as F
import pudb 
import itertools
import math

class Net(nn.Module):
    def __init__(self, textVocabLen):
        super(Net, self).__init__()
        embedding_dim = 50
        self.embed = nn.Embedding(textVocabLen, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, labelVocabLen)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(1)
        x = self.fc1(x)
        return F.log_softmax(x)

class SimpleCmd():
    def __init__(self, run_context):
        self.run_context = run_context

        self.criterion = nn.NLLLoss()
        self.encoder = SimpleEncodeModel(
            run_context.nl_vocab_size, run_context.std_word_size)
        self.predictProgram = PredictProgramModel(
            run_context.std_word_size, run_context.num_of_descriptions)
        self.all_modules = nn.ModuleList([self.encoder, self.predictProgram])

        self.optimizer = optim.Adam(self.all_modules.parameters())

    def train_step(self, batch):
        self.all_modules.train()
        self.optimizer.zero_grad()

        query, query_lengths = batch.nl
        ast = batch.command
        firstCommands = [a[0] for a in ast]
        expectedProgIndicies = self.run_context.make_choice_tensor(firstCommands)
        encodings = self.encoder(query)
        pred = self.predictProgram(encodings)
        loss = self.criterion(pred, expectedProgIndicies)
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def eval_step(self, batch):
        query, query_lengths = batch.nl
        ast = batch.command
        firstCommands = [a[0] for a in ast]
        expectedProgIndicies = self.run_context.make_choice_tensor(firstCommands)
        encodings = self.encoder(query)
        pred = self.predictProgram(encodings)
        print("pred ", pred, "gt", expectedProgIndicies)
        return pred.data.cpu(), expectedProgIndicies.data.cpu()

class SimpleEncodeModel(nn.Module):
    """Creates a one word description of whole sequence"""
    def __init__(self, nl_vocab_size, std_word_size):
        super(SimpleEncodeModel, self).__init__()

        # General stuff
        self.std_word_size = std_word_size

        # forward pass stuff
        self.embed = nn.Embedding(nl_vocab_size, std_word_size)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(1)
        return x

class PredictProgramModel(nn.Module):
    def __init__(self, std_word_size, num_of_programs):
        super(PredictProgramModel, self).__init__()

        # General stuff
        self.std_word_size = std_word_size
        self.chooseProgramFC = nn.Linear(std_word_size, num_of_programs)

    def forward(self, x):
        x = self.chooseProgramFC(x)
        return F.log_softmax(x)
