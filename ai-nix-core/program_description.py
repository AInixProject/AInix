"""This module contains utilities for handling program descriptions.
These are typically specified in a program.yaml that is packaged with program.
"""
import io
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def parse_program_description(program_description):
    """Parse the contents of a program.yaml"""
    data = yaml.safe_load(program_description)
    print(data)
    out_desc = AIPrgramDescription(
        name = data.get('name')
    )
    return out_desc

class AIPrgramDescription():
    """Describes an AI program. This is the deserialized instantiation
    that comes from a program.yaml"""
    def __init__(self, name, arguments = []):
        self.name = name
        self.arguments = arguments 
