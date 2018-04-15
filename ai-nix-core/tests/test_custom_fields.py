from custom_fields import Replacer, ReplacementGroup, Replacement, WeightedRandomChooser

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
    replacement = Replacement("foo", "boo", 1) 
    replacement2 = Replacement("moo", "cow", 1) 
    rg = ReplacementGroup('TEST', [replacement])
    rg2 = ReplacementGroup('TSET', [replacement])
    replacer = Replacer([rg])
    nl, cmd = replacer.strings_replace("[-[TEST]-] hello [-[TSET]-]", "[-[TEST]-] yo [-[TSET]-]")
    assert nl == "foo hello moo"
    assert cmd == "moo yo cow"
