from custom_fields import *
from collections import Counter
from unittest.mock import MagicMock, Mock


def test_replacer1():
    replacement = Replacement("foo", "boo", 1) 
    rg = ReplacementGroup('TEST', [replacement])
    replacer = Replacer([rg])
    nl, cmd = replacer.strings_replace("hello [-[TEST]-]", "run [-[TEST]-]")
    assert nl == "hello foo"
    assert cmd == "run boo"

def test_replacer2():
    replacement = Replacement("foo", "boo", 1) 
    rg = ReplacementGroup('TEST', [replacement])
    replacer = Replacer([rg])
    nl, cmd = replacer.strings_replace("hello [-[TEST]-] [-[2=TEST]-]", "run [-[TEST]-] [-[$2]-]")
    assert nl == "hello foo foo"
    assert cmd == "run boo boo"

def test_replacer3():
    """Test detecting multiple kinds of replacements"""
    replacement = Replacement("foo", "boo", 1) 
    replacement2 = Replacement("moo", "cow", 1) 
    rg = ReplacementGroup('TEST', [replacement])
    rg2 = ReplacementGroup('TSET', [replacement2])
    replacer = Replacer([rg, rg2])
    nl, cmd = replacer.strings_replace("[-[TEST]-] hello [-[TSET]-]", "[-[TEST]-] yo [-[TSET]-]")
    assert nl == "foo hello moo"
    assert cmd == "boo yo cow"

def test_replacer4():
    """Test sampleing"""
    replacement = Replacement("foo", "foo", 1) 
    replacement2 = Replacement("moo", "moo", 3) 
    rg = ReplacementGroup('TEST', [replacement, replacement2])
    replacer = Replacer([rg])
    samples = [replacer.strings_replace("hello [-[TEST]-]", "yo [-[TEST]-]") for i in range(1000)]
    nlres, cmdres = zip(*samples)
    nlcount = Counter(nlres)
    assert abs(nlcount["hello foo"] - 250) < 40
    assert abs(nlcount["hello moo"] - 750) < 40

def test_replacer5():
    """Test reusing same thing multiple times in cmd"""
    replacement = Replacement("foo", "bar", 1) 
    rg = ReplacementGroup('TEST', [replacement])
    replacer = Replacer([rg])
    nl, cmd = replacer.strings_replace("hello [-[TEST]-]", "run [-[TEST]-] [-[TEST]-]")
    assert nl == "hello foo"
    assert cmd == "run bar bar"

def test_replacer6():
    """Test numbered replacements"""
    replacement = Replacement("foo", "bar", 1) 
    rg = ReplacementGroup('TEST', [replacement])
    replacer = Replacer([rg])
    nl, cmd = replacer.strings_replace("hello [-[1=TEST]-] [-[2=TEST]-]", "run [-[$1]-] [-[$2]-]")
    assert nl == "hello foo foo"
    assert cmd == "run bar bar"

def test_replacer7():
    """Test replacement args"""
    replacement = Replacement("foo", "bar", 1) 
    rg = ReplacementGroup('TEST', [replacement])
    rg.sample_replacement = MagicMock(return_value=("foo", "bar"))
    replacer = Replacer([rg])
    nl, cmd = replacer.strings_replace("hello [-[TEST -t foo]-]", "run")
    rg.sample_replacement.assert_called_once_with(["TEST", '-t', "foo"])

def test_replacer8():
    """Test replacement args with vars"""
    replacement = Replacement("foo", "bar", 1) 
    rg = ReplacementGroup('TEST', [replacement])
    rg.sample_replacement = MagicMock(return_value=("foo", "bar"))
    replacer = Replacer([rg])
    nl, cmd = replacer.strings_replace("hello [-[1=TEST -t foo]-]", "run")
    rg.sample_replacement.assert_called_once_with(["TEST", '-t', "foo"])


######################################################################

def test_copy_tok():
    """Test replacement args with vars"""
    vocab = Mock()
    exunk = "AnUnk"
    vocab.freqs.__getitem__ = lambda self, word: 0 if word == exunk else 10
    test_seq = [constants.SOS, "hello", constants.SPACE, "there", constants.SPACE, exunk, constants.EOS]
    field = Mock()
    field.vocab = vocab
    example = NLExample(test_seq, field)
    expected_mod = [constants.SOS, "hello", constants.SPACE, "there", constants.SPACE, 
            constants.COPY_TOKENS[0], exunk, constants.EOS]
    assert example.mod_text == expected_mod
    #assert example.subsequence_to_copy[(exunk,)] == constants.COPY_TOKENS[0]
    # try and replace
    val_with_cp = example.insert_copies(["foo", exunk])
    assert val_with_cp == ["foo", constants.COPY_TOKENS[0]]




