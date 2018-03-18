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

class StoreTrue():
    pass

class Argument():
    """Represents an argument in a AIProgramDescription"""
    def __init__(self, name, argtype, shorthand = None):
        self.name = name
        self.shorthand = shorthand
        self.argtype = argtype
        pass

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

    valid_top_level_keys = ['name', 'arguments']
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

    def get_examples(self):
        return self.examples
