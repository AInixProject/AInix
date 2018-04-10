from __future__ import (absolute_import, division,
                        print_function, unicode_literals) 
import torch 
from torch import Tensor, nn, optim
from torch.autograd import Variable
import torchtext
import torch.nn.functional as F
import pudb 
import itertools
import math
from cmd_parse import ProgramNode, ArgumentNode
import constants

class SimpleCmd():
    def __init__(self, run_context):
        self.run_context = run_context

        # Build the argument data for this model
        encodeType = torch.cuda.FloatTensor if run_context.use_cuda else torch.FloatTensor
        all_arg_params = []
        for prog in run_context.descriptions:
            all_top_v = []
            if prog.arguments is None:
                continue
            for arg in prog.arguments:
                topv = Variable(torch.randn(run_context.std_word_size).type(encodeType), requires_grad = True)
                all_top_v.append(topv)
                all_arg_params.append(topv)
                arg.model_data = {"top_v": topv} 
            prog.model_data_grouped = {"top_v": torch.stack(all_top_v) if all_top_v else None}

        # Create layers
        self.criterion = nn.NLLLoss()
        self.encoder = SimpleEncodeModel(
            run_context.nl_vocab_size, run_context.std_word_size)
        self.decoder = SimpleDecoderModel(
            run_context.std_word_size, run_context.nl_vocab_size)
        self.predictProgram = PredictProgramModel(
            run_context.std_word_size, run_context.num_of_descriptions)
        self.all_modules = nn.ModuleList([self.encoder, self.decoder, self.predictProgram])
        if run_context.use_cuda:
            self.all_modules.cuda()


        self.optimizer = optim.Adam(list(self.all_modules.parameters()) + all_arg_params)

    def train_step(self, engine, batch):
        self.all_modules.train()
        self.optimizer.zero_grad()

        query, query_lengths = batch.nl
        ast = batch.command
        firstCommands = [a[0] for a in ast]
        expectedProgIndicies = self.run_context.make_choice_tensor(firstCommands)
        encodings = self.encoder(query)
        pred = self.predictProgram(encodings)
        loss = self.criterion(pred, expectedProgIndicies)
        
        for i, firstCmd in enumerate(firstCommands):
            # Go through and predict each argument present or not
            if len(firstCmd.program_desc.arguments) > 0:
                argDots = torch.mv(firstCmd.program_desc.model_data_grouped['top_v'], encodings[i])
                argLoss = F.binary_cross_entropy_with_logits(argDots, firstCmd.arg_present_tensor,
                        size_average = False)
                loss += argLoss
            # Go through and predict the value of present nodes
            for arg_node in firstCmd.arguments:
                if arg_node.present and arg_node.value is not None:
                    argtype = arg_node.arg.argtype
                    parsed_value = argtype.parse_value(arg_node.value, self.run_context)
                    loss += argtype.train_value(encodings[i], parsed_value, self.run_context, self)

        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def eval_step(self, engine, batch):
        self.all_modules.eval()
        query, query_lengths = batch.nl
        ast = batch.command
        firstCommands = [a[0] for a in ast]
        expectedProgIndicies = self.run_context.make_choice_tensor(firstCommands)
        encodings = self.encoder(query)
        predPrograms = self.predictProgram(encodings)
        vals, predProgramsMaxs =  predPrograms.max(1)
        #print("pred ", predPrograms, "gt", expectedProgIndicies)

        # Go through and predict each argument
        pred = [[] for b in firstCommands]
        predProgramDescs = [self.run_context.descriptions[m.data[0]] for m in predProgramsMaxs]
        for i, predProgragmD in enumerate(predProgramDescs):
            setArgs = []
            if len(predProgragmD.arguments) > 0:
                argDots = torch.mv(predProgragmD.model_data_grouped['top_v'], encodings[i])
                argSig = F.sigmoid(argDots)
                for aIndex, arg in enumerate(predProgragmD.arguments):
                    thisArgPredicted = argSig[aIndex].data[0] > 0.5
                    if thisArgPredicted and arg.argtype.requires_val:
                        predVal = arg.argtype.eval_value(encodings[i], self.run_context, self)
                    else:
                        predVal = thisArgPredicted

                    setArgs.append(ArgumentNode(arg, thisArgPredicted, predVal))

            pred[i].append(ProgramNode(predProgragmD, setArgs, self.run_context.use_cuda))

        return pred, batch.command

    def std_decode_train(self, encodeing, run_context, expected_tensor):
        decoder_input = Variable(torch.LongTensor([[run_context.nl_field.vocab.stoi[constants.SOS]]]))  
        decoder_input = decoder_input.cuda() if run_context.use_cuda else decoder_input

        loss = 0
        criterion = nn.NLLLoss()

        decoder_hidden = encodeing.view(1,1,-1) # double unsqueeze
        target_length = expected_tensor.size()[1]
        for di in range(1, target_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, expected_tensor[0][di])
            decoder_input = expected_tensor[0][di]  # Teacher forcing
        return loss
    
    def std_decode_eval(self, encodeing, run_context, max_length = 5):
        decoder_input = Variable(torch.LongTensor([[run_context.nl_field.vocab.stoi[constants.SOS]]]))  
        decoder_input = decoder_input.cuda() if run_context.use_cuda else decoder_input

        decoded_words = [constants.SOS]
        decoder_hidden = encodeing.view(1,1,-1) # double unsqueeze
        for di in range(max_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoded_words.append(run_context.nl_field.vocab.itos[ni])
            if decoded_words[-1] == constants.EOS:
                decoded_words = decoded_words[1:-1]
                break
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if run_context.use_cuda else decoder_input

        return decoded_words



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
        return F.log_softmax(x, dim=1)

class SimpleDecoderModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(SimpleDecoderModel, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
