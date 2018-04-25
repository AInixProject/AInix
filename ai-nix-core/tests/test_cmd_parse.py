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

cValArg = program_description.Argument("c", "Stringlike")

valprog = program_description.AIProgramDescription(
    name = "valprog",
    arguments = [aArg, bArg, cValArg, posArg] 
)


def test_noarg_parse():
    parser = CmdParser([noArgs])
    out = parser.parse("noarg")
    assert len(out) == 2
    assert out[0].program_desc == noArgs
    assert len(out[0].arguments) == 0

def test_onearg_parse():
    parser = CmdParser([noArgs, onearg])
    out = parser.parse("onearg -a")
    assert len(out) == 2
    assert out[0].program_desc == onearg
    assert len(out[0].arguments) == 1
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[0].present == True
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1])))

def test_onearg_missing_parse():
    parser = CmdParser([noArgs, onearg])
    out = parser.parse("onearg")
    assert len(out) == 2
    assert out[0].program_desc == onearg
    assert len(out[0].arguments) == 1
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[0].present == False

def test_two_arg():
    parser = CmdParser([noArgs, onearg, twoarg])
    out = parser.parse("twoarg -a")
    assert len(out) == 2
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == False
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,0])))

    out = parser.parse("twoarg -b")
    assert len(out) == 2
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[0].present == False
    assert out[0].arguments[1].present == True
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([0,1])))

    out = parser.parse("twoarg -a -b")
    assert len(out) == 2
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[1].arg == bArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1])))

    out = parser.parse("twoarg -ab")
    assert len(out) == 2
    assert out[0].program_desc == twoarg
    assert len(out[0].arguments) == 2
    assert out[0].arguments[0].arg == aArg
    assert out[0].arguments[1].arg == bArg
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1])))

    out = parser.parse("twoarg -ba")
    assert len(out) == 2
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

def test_val_arg():
    parser = CmdParser([valprog])
    out = parser.parse("valprog -c foo")
    assert out[0].arguments[0].present == False
    assert out[0].arguments[1].present == False
    assert out[0].arguments[2].present == True
    assert out[0].arguments[3].present == False
    assert out[0].arguments[2].value == "foo"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([0,0,1,0])))

    out = parser.parse("valprog -a -c bar main.c")
    assert out[0].arguments[2].value == "bar"
    assert out[0].arguments[3].value == "main.c"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,0,1,1])))

    out = parser.parse("valprog -ba -c \"foo bar\" main.c")
    assert out[0].arguments[2].value == "foo bar"
    assert out[0].arguments[3].value == "main.c"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1,1,1])))

def test_val_combined():
    """A test of whether argument value parsing works when there is no
    space between the flag and the value"""
    parser = CmdParser([valprog])
    out = parser.parse("valprog -cfoo")
    assert out[0].arguments[2].value == "foo"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([0,0,1,0])))

def test_pipe():
    """A test of whether argument value parsing works when there is no
    space between the flag and the value"""
    parser = CmdParser([posandtwo, valprog])
    out = parser.parse("posandtwo | valprog")
    assert isinstance(out[0], ProgramNode)
    assert out[0].program_desc == posandtwo
    assert isinstance(out[1], PipeNode)
    assert isinstance(out[2], ProgramNode)
    assert out[2].program_desc == valprog
    assert isinstance(out[3], EndOfCommandNode)

def test_pipe2():
    """A test of whether argument value parsing works when there is no
    space between the flag and the value"""
    parser = CmdParser([posandtwo, valprog])
    out = parser.parse("posandtwo | valprog | posandtwo")
    assert isinstance(out[0], ProgramNode)
    assert isinstance(out[1], PipeNode)
    assert isinstance(out[2], ProgramNode)
    assert isinstance(out[3], PipeNode)
    assert isinstance(out[4], ProgramNode)
    assert isinstance(out[5], EndOfCommandNode)

###
# Test correct as_shell_string
###

def test_as_shell_string1():
    parser = CmdParser([posandtwo])
    out = parser.parse("posandtwo -a -b main.c")
    assert out[0].as_shell_string() == "posandtwo -a -b main.c"

    out = parser.parse("posandtwo -ab main.c")
    assert out[0].as_shell_string() == "posandtwo -a -b main.c"

    out = parser.parse("posandtwo -a main.c")
    assert out[0].as_shell_string() == "posandtwo -a main.c"

    out = parser.parse("posandtwo -b -a main.c")
    assert out[0].as_shell_string() == "posandtwo -a -b main.c"

def test_as_shell_string2():
    parser = CmdParser([valprog])
    out = parser.parse("valprog -c foo")
    assert out[0].as_shell_string() == "valprog -c foo"

def test_as_shell_pipe():
    parser = CmdParser([valprog, posandtwo])
    out = parser.parse("valprog -c foo | posandtwo")
    assert out.as_shell_string() == "valprog -c foo | posandtwo"


@pytest.mark.skip(reason="will need to fix this when implement custom parser")
def test_as_shell_string_quotes():
    parser = CmdParser([valprog])
    out = parser.parse("valprog -c \"foo bar\" main.c")
    assert out[0].as_shell_string() == "valprog -c \"foo bar\" main.c"
