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
