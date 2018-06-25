from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import torch
from torch.autograd import Variable
class RunContext():
    """This class stores data useful during the run of models"""
    def __init__(self, std_word_size, nl_field, cmd_field, program_descriptions, use_cuda, batch_size = 1, debug = False, quiet_mode = False):
        self.std_word_size = std_word_size
        self.small_word_size = int(std_word_size / 4)
        self.nl_field = nl_field
        self.cmd_field = cmd_field
        self.descriptions = program_descriptions
        self.num_of_descriptions = len(self.descriptions)
        self.nl_vocab_size = len(nl_field.vocab.itos)
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.debug = debug
        self.quiet_mode = quiet_mode
        self._setdevice()

        # Set the program index on all programs.
        # TODO (dngros): figure out a way to do this without mutate program_descriotiosn
        for i, prog in enumerate(program_descriptions):
            prog.program_index = i

    def fill_argument_data(self, argument_builder):
        """Used to initiallize the model data for all arguments.
            argument_builder (callable): a function that takes in a argument and returns
                                         a object to set that argument model data to
        """
        # Create argument model data
        for prog in self.descriptions:
            if prog.arguments is None:
                continue
            for arg in prog.arguments:
                arg.model_data = argument_builder(arg)


    def make_choice_tensor(self, descs):
        """Makes a tensor of the proper program_index for a batch of descriptions"""
        encodeType = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        indicies = [d.program_desc.program_index for d in descs] 
        return Variable(encodeType(indicies), requires_grad = False)

    def _setdevice(self):
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def __getstate__(self):
        odict = self.__dict__
        # device doesnt like being serialized
        del odict['device']
        return odict

    def __setstate__(self, sdict):
        self.__dict__ = sdict
        self._setdevice()
        
