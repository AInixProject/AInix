import pytest
import program_description
from program_description import ProgramParseError
import sys, os
from io import StringIO

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

def test_parse_program_description():
    with open('./tests/test-program.yaml', 'r') as f:
        desc = program_description.load(f)
    assert desc.name == "test"

def test_invalid_top_level():
    testFile = \
    """
    name: foo
    badTopLevel: sdfasdf
    """
    with pytest.raises(ProgramParseError):
        program_description.load(testFile) 

def test_invalid_top_level2():
    """missing name"""
    testFile = \
    """
    arguments: sdfasdf
    """
    with pytest.raises(ProgramParseError):
        program_description.load(testFile) 

def test_invalid_top_level3():
    """non-list arguements"""
    testFile = \
    """
    name: foo
    arguments:
        foo: this shouldnt be here. It should be a list
    """
    with pytest.raises(ProgramParseError):
        program_description.load(testFile) 

@pytest.mark.skip(reason="need to fix the footype argument parsing")
def test_invalid_top_level4():
    """non-list arguements"""
    testFile = \
    """
    name: foo
    arguments:
        - FooType:
            name: "--foo"
        - FooType: sdf
          WhatIsThis: sdf
    """
    with pytest.raises(ProgramParseError):
        program_description.load(testFile) 
