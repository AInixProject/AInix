import pytest
from cmd_parse import *
import program_description
import torch
from torch.autograd import Variable

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

posArg = program_description.Argument("aposarg", "Stringlike", position = 0)
justpos = program_description.AIProgramDescription(
    name = "justpos",
    arguments = [posArg] 
)

posandtwo = program_description.AIProgramDescription(
    name = "posandtwo",
    arguments = [aArg, bArg, posArg] 
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
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1])))

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
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,0])))

    out = parser.parse("twoarg -b")
    assert len(out) == 1
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[0].present == False
    assert out[0].arguments[1].present == True
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([0,1])))

    out = parser.parse("twoarg -a -b")
    assert len(out) == 1
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[1].arg == bArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1])))

    out = parser.parse("twoarg -ab")
    assert len(out) == 1
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[1].arg == bArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1])))

    out = parser.parse("twoarg -ba")
    assert len(out) == 1
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[1].arg == bArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1])))

def test_just_pos():
    parser = CmdParser([justpos])
    out = parser.parse("justpos main.c")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[0].value == "main.c"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1])))

    out = parser.parse("justpos main.c ~/sdf/")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[0].value == "main.c ~/sdf/"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1])))

def test_pos_and_args():
    parser = CmdParser([posandtwo])
    out = parser.parse("posandtwo -a -b main.c")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arguments[2].present == True
    assert out[0].arguments[2].value == "main.c"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1,1])))

    out = parser.parse("posandtwo -a -b")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arguments[2].present == False
    assert out[0].arguments[2].value == None
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1,0])))
