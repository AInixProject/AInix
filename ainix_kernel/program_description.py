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
from ainix_kernel import constants
import hjson
from ainix_kernel import arg_types

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

class Argument():
    """Represents an argument in a AIProgramDescription"""
    def __init__(self, name, argtype, shorthand = None, position = None, 
        required = False, long_single_dash = False):
        self.name = name
        self.shorthand = shorthand
        self.type_name = argtype
        # Figure out arg type
        if argtype not in arg_types.__dict__:
            raise ValueError("unknown arg type ", argtype)
        if not issubclass(arg_types.__dict__[argtype], arg_types.ArgumentType):
            raise ValueError("Attempt to assign argtype of non-argtype", argtype)
        self.argtype = arg_types.__dict__[argtype]()

        self.position = position
        self.required = required
        self.long_single_dash = long_single_dash

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
        _verify_keys(data.keys(), valid_keys=['name','type','shorthand','position', 'long_single_dash'], required_keys=['name','type'])
        # TODO (DNGros) I dont like the deplicate code with specifying defaults. Could
        # get out of sync and be an annoying bug. Look into.
        return cls(data['name'], data['type'], data.get('shorthand', None), 
                position = data.get('position',None), long_single_dash = data.get('long_single_dash', False))

def load(program_description):
    """Parse the contents of a progdesc.hjson into an AIProgramDescription"""
    data = hjson.loads(program_description)

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
        self.examples = examples

        # Check to make sure valid positional arguments
        positional_args = [arg for arg in arguments if arg.position is not None]
        positional_args.sort(key = lambda a: a.position)
        if sum([1 if a.argtype.is_multi_word else 0 for a in positional_args]) > 1:
            raise ValueError("Cannont create description with more than one \
                    multiword postional args. Leads to ambigious parsing. Name:", self.name)
        self.smallest_pos_number = None
        if positional_args:
            actual_pos_numbers = tuple([a.position for a in positional_args])
            smallest_pos_number = actual_pos_numbers[0]
            self.smallest_pos_number = smallest_pos_number
            largest_expected_pos = len(positional_args) + smallest_pos_number
            if largest_expected_pos < 0:
                raise ValueError("Description", self.name, 
                        " has bad non-sequential positional arguments", actual_pos_numbers, 
                        ". Too many negative values")
            expectedActualPosNumbers = tuple(range(smallest_pos_number, 
                largest_expected_pos))
            if actual_pos_numbers != expectedActualPosNumbers:
                raise ValueError("Description", self.name, 
                        " has non-sequential positional arguments", actual_pos_numbers, 
                        ". Expect", expectedActualPosNumbers)
        self.positional_args = positional_args
        self.has_pos_args_before_flags = positional_args and positional_args[0].position < 0
        # TODO (dngros): this should check if there are duplicate arguments with the same name

        # Program index is set during training to catagorize programs
        self.program_index = None
        self.model_data_grouped = None

    def get_examples(self):
        return self.examples

    def get_example_tuples(self):
        out = []
        for example in self.examples:
            for x in example['lang']:
                out.append((x, example['cmd'][0]))
        return out
