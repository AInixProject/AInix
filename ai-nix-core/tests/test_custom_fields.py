from custom_fields import Replacer, ReplacementGroup, Replacement, WeightedRandomChooser
from collections import Counter

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
    nl, cmd = replacer.strings_replace("hello [-[TEST]-] [-[2.TEST]-]", "run [-[TEST]-] [-[2.TEST]-]")
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
    assert abs(nlcount["hello foo"] - 250) < 30
    assert abs(nlcount["hello moo"] - 750) < 30

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
    nl, cmd = replacer.strings_replace("hello [-[1.TEST]-] [-[2.TEST]-]", "run [-[1.TEST]-] [-[2.TEST]-]")
    assert nl == "hello foo foo"
    assert cmd == "run bar bar"
