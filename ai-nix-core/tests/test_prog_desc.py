import pytest
import program_description
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

def test_parse_program_description():
    with open('./test-program.yaml', 'r') as f:
        desc = program_description.parse_program_description(f)
    assert desc.name == "test"
