from collections import Counter
from unittest.mock import MagicMock

import pytest

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.stringparser import StringParser, AstUnparser
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.specialtypes import allspecials
from ainix_kernel.training.augmenting.replacers import *


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
    samples = [replacer.strings_replace("hello [-[TEST]-]", "yo [-[TEST]-]") for _ in range(1000)]
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
    nl, cmd = replacer.strings_replace("hello [-[1=TEST -t foo]-]", "run")
    rg.sample_replacement.assert_called_once_with(["TEST", '-t', "foo"])


def test_replacer8():
    """Test replacement args with vars"""
    replacement = Replacement("foo", "bar", 1)
    rg = ReplacementGroup('TEST', [replacement])
    rg.sample_replacement = MagicMock(return_value=("foo", "bar"))
    replacer = Replacer([rg])
    nl, cmd = replacer.strings_replace("hello [-[1=TEST -t foo]-]", "run [-[$1]-]")
    rg.sample_replacement.assert_called_once_with(["TEST", '-t', "foo"])


def test_replacer_bad():
    """Test reusing same thing multiple times in cmd"""
    replacement = Replacement("foo", "bar", 1)
    rg = ReplacementGroup('TEST', [replacement])
    replacer = Replacer([rg])
    with pytest.raises(ReplacementError):
        nl, cmd = replacer.strings_replace("hello [-[TEST]-]", "run [-[BLOOP]-]")


def test_replacer_no_group():
    """Test reusing same thing multiple times in cmd"""
    replacement = Replacement("foo", "bar", 1)
    rg = ReplacementGroup('TEST', [replacement])
    replacer = Replacer([rg])
    with pytest.raises(ReplacementError):
        _ = replacer.create_replace_sampling("hello [-[BAD]-]")


def test_file_replacer():
    replacements = Replacement.from_tsv("../../../training/data/FILENAME.tsv")
    tc = TypeContext()
    loader = TypeContextDataLoader(tc, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(tc)
    tc.finalize_data()
    parser = StringParser(tc)
    unparser = AstUnparser(tc)

    for repl in replacements:
        x, y = repl.get_replacement()
        assert x == y
        ast = parser.create_parse_tree(x, "Path")
        result = unparser.to_string(ast)
        assert result.total_string == x


def test_dir_replacer():
    replacements = Replacement.from_tsv("../../../training/data/DIRNAME.tsv")
    tc = TypeContext()
    loader = TypeContextDataLoader(tc, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(tc)
    tc.finalize_data()
    parser = StringParser(tc)
    unparser = AstUnparser(tc)
    fails = []
    for repl in replacements:
        x, y = repl.get_replacement()
        assert x == y
        ast = parser.create_parse_tree(x, "Path")
        result = unparser.to_string(ast)
        if result.total_string != x:
            fails.append((x, result.total_string))
    assert not fails


