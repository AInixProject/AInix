import pytest
from ainix_kernel.cmd_parse import *
from ainix_kernel import program_description
import torch
from torch.autograd import Variable

# Define a bunch of args and programs
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

singleWordArg1 = program_description.Argument("bposarg", "SingleFile", position = 0)
singleWordArg2 = program_description.Argument("bposarg", "SingleFile", position = 1)
twosingleword = program_description.AIProgramDescription(
    name = "twosingleword",
    arguments = [singleWordArg1, singleWordArg2] 
)

multiWordArg = program_description.Argument("bposarg", "FileList", position = 0)
twowmulti = program_description.AIProgramDescription(
    name = "twowmulti",
    arguments = [multiWordArg, singleWordArg2] 
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
    assert out[0].arguments[2].value == '"foo bar"'
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
    """Test whether can parse a pipe"""
    parser = CmdParser([posandtwo, valprog])
    out = parser.parse("posandtwo | valprog")
    assert isinstance(out[0], ProgramNode)
    assert out[0].program_desc == posandtwo
    assert isinstance(out[1], PipeNode)
    assert isinstance(out[2], ProgramNode)
    assert out[2].program_desc == valprog
    assert isinstance(out[3], EndOfCommandNode)

def test_pipe2():
    """Test whether can parse several pipes"""
    parser = CmdParser([posandtwo, valprog])
    out = parser.parse("posandtwo | valprog | posandtwo")
    assert isinstance(out[0], ProgramNode)
    assert isinstance(out[1], PipeNode)
    assert isinstance(out[2], ProgramNode)
    assert isinstance(out[3], PipeNode)
    assert isinstance(out[4], ProgramNode)
    assert isinstance(out[5], EndOfCommandNode)

def test_two_single_word_pos():
    parser = CmdParser([twosingleword])
    out = parser.parse("twosingleword foo bar")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arguments[0].value == "foo"
    assert out[0].arguments[1].value == "bar"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1])))
    assert out[0].as_shell_string() == "twosingleword foo bar"

def test_multi_word_pos():
    parser = CmdParser([twowmulti])
    out = parser.parse("twowmulti why hello there bar")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[1].present == True
    assert out[0].arguments[0].value == "why hello there"
    assert out[0].arguments[1].value == "bar"
    assert out[0].arg_present_tensor.equal(Variable(torch.FloatTensor([1,1])))
    assert out[0].as_shell_string() == "twowmulti why hello there bar"

# TODO (dngros): should really add test all the failure cases of positional args
# TODO (dngros): need to handle non-required positional word parsing (like touch)

oneDashArg = program_description.Argument("name", "Stringlike", long_single_dash = True)
dashtest = program_description.AIProgramDescription(
    name = "dashtest",
    arguments = [oneDashArg] 
)

def test_dash_arg():
    """A test to see if can handle find-like single dash for long args"""
    parser = CmdParser([dashtest])
    out = parser.parse("dashtest -name foo")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[0].value == "foo"
    assert out[0].as_shell_string() == "dashtest -name foo"

oneDashArg = program_description.Argument("name", "Stringlike", long_single_dash = True)
typeArg = program_description.Argument("type_name", "Stringlike", long_single_dash = True)
findRoot = program_description.Argument("findroot", "SingleFile", position = -1)
findlike = program_description.AIProgramDescription(
    name = "findlike",
    arguments = [oneDashArg, findRoot, typeArg] 
)

def test_findlike():
    """A test to see if can handle find-like single dash for long args"""
    parser = CmdParser([findlike])
    out = parser.parse("findlike . -name foo")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[0].value == "foo"
    assert out[0].arguments[1].present == True
    assert out[0].arguments[1].value == "."
    assert out[0].as_shell_string() == "findlike . -name foo"

def test_duplicate_flags():
    """Dont expect duplicate flags unless told."""
    parser = CmdParser([noArgs, onearg])
    with pytest.raises(CmdParseError):
        out = parser.parse("onearg -a -a")

def test_unrecognized_flag():
    """If use a flag that is not specified, should error"""
    parser = CmdParser([noArgs, onearg])
    with pytest.raises(CmdParseError):
        out = parser.parse("onearg -a -b")

def test_extra_positional():
    """Make sure recognizes extra words at the end"""
    parser = CmdParser([noArgs, onearg])
    with pytest.raises(CmdParseError):
        out = parser.parse("onearg -a thisShouldNotBeHere")

def test_unrecognized_find_flag():
    parser = CmdParser([findlike])
    with pytest.raises(CmdParseError):
        out = parser.parse("findlike foo -name foo -size 1")

def test_no_root():
    parser = CmdParser([findlike])
    out = parser.parse("findlike -name foo -type_name f")
    assert out[0].arguments[0].present == True
    assert out[0].arguments[0].value == "foo"
    assert out[0].arguments[1].present == False
    assert out[0].arguments[2].present == True
    assert out[0].arguments[2].value == "f"

def test_unrecognized_extra_word():
    # NOTE: this test shouldnt be necessary once actually works correctly
    parser = CmdParser([findlike])
    with pytest.raises(CmdParseError):
        out = parser.parse("findlike foo ! -name foo")

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

def test_as_shell_string_quotes():
    parser = CmdParser([valprog])
    out = parser.parse("valprog -c \"foo bar\" main.c")
    assert out[0].as_shell_string() == "valprog -c \"foo bar\" main.c"
