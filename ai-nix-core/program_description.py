"""This module contains utilities for handling program descriptions.
These are typically specified in a program.yaml that is packaged with program.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import io
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pudb
import constants
import hjson

class ProgramParseError(Exception):
    pass

def _verify_keys(keys, valid_keys = None, required_keys = None):
    """Used while parsing yamls to verify loaded dictionary has expected keys"""
    if valid_keys is not None:
        if not set(keys).issubset(set(valid_keys)):
            raise ProgramParseError("Unexpected Top level attributes", 
                set(keys) - set(valid_keys))
    if required_keys is not None:
        if not set(keys).issuperset(set(required_keys)):
            raise ProgramParseError("Missing Required Top level attributes", 
                set(required_keys) - set(keys))

class ArgumentType():
    def __init__(self):
        self.model_data = None
        self.requires_value = False

class StoreTrue(ArgumentType):
    def as_shell_string(self, value):
        return ''

class Stringlike(ArgumentType):
    def __init__(self):
        super(Stringlike, self).__init__()
        self.requires_value = True

    def parse_value(self, value, run_context, copyfromexample, is_eval = False):
        """Takes in a value and converts it to friendly representation"""
        #preproc = run_context.nl_field.preprocess(value)
        #return run_context.nl_field.process([preproc], 
        #        device = 0 if run_context.use_cuda else -1,
        #        train = not is_eval)[0] # take 0th since a tuple with length

        preproc = run_context.nl_field.preprocess(value)
    
        # hacky copy
        for sequence, copytoken in copyfromexample.subsequence_to_copy.items():
            if len(sequence) > 1:
                raise NotImplementedError("havent made actual subsequencs work", sequence)
            lookVal = sequence[0]
            if lookVal in preproc:
                preproc[preproc.index(lookVal)] = copytoken

        padded = run_context.nl_field.pad([preproc])
        (tensor, lengths) = run_context.nl_field.numericalize(padded, device=0 if run_context.use_cuda else -1, train=not is_eval)
        return tensor

    def train_value(self, encoding, expected_value, run_context, meta_model):
        return meta_model.std_decode_train(encoding, run_context, expected_value)

    def eval_value(self, encoding, run_context, meta_model, copyfromexample):
        predSequence = meta_model.std_decode_eval(encoding, run_context)
        copied = []
        for p in predSequence:
            if p in constants.COPY_TOKENS:
                if p in copyfromexample.copy_to_sequence:
                    copied += list(copyfromexample.copy_to_sequence[p])
                else:
                    copied.append(p)
            else:
                copied.append(p)

        return " ".join(copied)

    def as_shell_string(self, value):
        return value

class Argument():
    """Represents an argument in a AIProgramDescription"""
    def __init__(self, name, argtype, shorthand = None, position = None, required = False):
        self.name = name
        self.shorthand = shorthand
        self.type_name = argtype
        if argtype == "StoreTrue":
            self.argtype = StoreTrue()
        elif argtype == "Stringlike":
            self.argtype = Stringlike()
        else:
            raise ValueError("unknown arg type ", argtype)
        self.position = position
        self.required = required

        # stores argument specific data for models
        self.model_data = None

    def as_shell_string(self, value):
        """Gets the shell string version of this Argument given the value"""
        out = ""
        if self.position is None:
            if self.name is not None:
                out += "--" if self.shorthand else "-"
                out += self.name
            elif self.shorthand is not None:
                out += "-" + self.shorthand
        argval = self.argtype.as_shell_string(value)
        if argval != '' and self.position is None:
            out += " "
        out += argval
        return out

        

    @classmethod
    def load(cls, data):
        """Instantiate an Argument from a dict that came from loading a yaml"""
        _verify_keys(data.keys(), valid_keys=['name','type','shorthand'], required_keys=['name','type'])
        return cls(data['name'], data['type'], data.get('shorthand', None))

def load(program_description):
    """Parse the contents of a progdesc.hjson into an AIProgramDescription"""
    data = hjson.loads(program_description)
    print(data)

    valid_top_level_keys = ['name', 'arguments', 'examples']
    required_top_level_keys = ['name']
    _verify_keys(data.keys(), valid_top_level_keys, required_top_level_keys)
     
    # Parse the arguments
    arg_data = data.get('arguments', [])
    if not isinstance(arg_data, list):
        raise ProgramParseError("Expected arguments to be a list")
    args = []
    for argD in arg_data:
        args.append(Argument.load(argD))
             
    # Construct the final output
    out_desc = AIProgramDescription(
        name = data.get('name'),
        arguments = args,
        examples = data.get('examples', [])
    )
    return out_desc

class AIProgramDescription():
    """Describes an AI program. This is the deserialized instantiation
    that comes from a program.yaml"""
    def __init__(self, name, arguments = [], examples = []):
        self.name = name
        self.arguments = arguments 
        self.examples = []
        # Program index is set during training to catagorize programs
        self.program_index = None
        self.model_data_grouped = None

    def get_examples(self):
        return self.examples
