import pytest
from cmd_parse import *
import program_description


noArgs = program_description.AIProgramDescription(
    name = "noarg"
)

aArg = program_description.Argument("a", "StoreTrue")
onearg = program_description.AIProgramDescription(
    name = "onearg",
    arguments = [aArg] 
)

bArg = program_description.Argument("b", "StoreTrue")
twoarg = program_description.AIProgramDescription(
    name = "twoarg",
    arguments = [aArg, bArg] 
)

def test_noarg_parse():
    parser = CmdParser([noArgs])
    out = parser.parse("noarg")
    assert len(out) == 1
    assert out[0].program_desc == noArgs
    assert len(out[0].arguments) == 0

def test_onearg_parse():
    parser = CmdParser([noArgs, onearg])
    out = parser.parse("onearg -a")
    assert len(out) == 1
    assert out[0].program_desc == onearg
    assert len(out[0].arguments) == 1
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[0].present == True

def test_onearg_missing_parse():
    parser = CmdParser([noArgs, onearg])
    out = parser.parse("onearg")
    assert len(out) == 1
    assert out[0].program_desc == onearg
    assert len(out[0].arguments) == 1
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[0].present == False

def test_two_arg():
    parser = CmdParser([noArgs, onearg, twoarg])
    out = parser.parse("twoarg -a")
    assert len(out) == 1
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == False

    out = parser.parse("twoarg -a -b")
    assert len(out) == 1
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[1].arg == bArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True

    out = parser.parse("twoarg -ab")
    assert len(out) == 1
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[1].arg == bArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True

    out = parser.parse("twoarg -ba")
    assert len(out) == 1
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[1].arg == bArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
