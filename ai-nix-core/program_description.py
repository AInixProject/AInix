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
        self.requires_val = False

class StoreTrue(ArgumentType):
    pass

class Stringlike(ArgumentType):
    def __init__(self):
        super(Stringlike, self).__init__()
        self.requires_val = True

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

class Argument():
    """Represents an argument in a AIProgramDescription"""
    def __init__(self, name, argtype, shorthand = None, position = None, required = False):
        self.name = name
        self.shorthand = shorthand
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

    @classmethod
    def load(cls, data):
        """Instantiate an Argument from a dict that came from loading a yaml"""
        if len(data) is not 1:
            raise ProgramParseError(
                "Invalid Argument. There should only be one type specified", data)
        print("Argument", list(data.items()))
        type_name, keys = list(data.items())[0]
        _verify_keys(keys, ['name'], None)
        return cls(keys['name'], type_name, keys.get('shorthand', None))

def load(program_description):
    """Parse the contents of a program.yaml into an AIProgramDescription"""
    data = yaml.safe_load(program_description)
    print(data)

    valid_top_level_keys = ['name', 'arguments', 'training-data']
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
        arguments = args
    )
    return out_desc

class AIProgramDescription():
    """Describes an AI program. This is the deserialized instantiation
    that comes from a program.yaml"""
    def __init__(self, name, arguments = [], in_data_file = None, out_data_file = None):
        self.name = name
        self.arguments = arguments 
        self.in_data_file = in_data_file
        self.out_data_file = out_data_file
        # Program index is set during training to catagorize programs
        self.program_index = None
        self.model_data_grouped = None

    def get_examples(self):
        return self.examples
