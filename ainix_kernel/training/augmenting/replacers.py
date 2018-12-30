"""Replacers can be used to augment a dataset.

They look like [-[REPLACER_NAME]-] in the x and y which will replaced with
randomly sampled values.
For example:
    x: "my name is [-[NAME]-]"
    y: "hello [-[NAME]-]"
could be replaced with a sampling of first names.

If you have multiple replacers of the same type in one (x, y) pair, you should
assign them numbers to disambiguate them.
For example
    x: "this is [-[1=NAME]-] and [-[2=NAME]-]"
    y: "hello [-[$1]-] and [-[$2]-]"
This will properly fill in whatever was the first sampled value in the first spot
and the second sampled value in the second value

Code was written for replacer arguments like
    [-[NUMBER --lessthan 15]-]
which could be used to give parameters to the replacers in case the user wanted
somehow use that, but this isn't actually really used or builtout.
"""
import csv
import re
from typing import List

from ainix_kernel.util.sampling import WeightedRandomChooser


class ReplacementError(ValueError):
    pass


class Replacer:
    """A class that will fill in the values """
    pattern = r"\[-\[.+?\]-\]"
    reg = re.compile(pattern)

    def __init__(self, types: List['ReplacementGroup']):
        self.types = types
        self.nameToTypes = {t.name: t for t in types}

    def strings_replace(self, nl, cmd):
        """The main method used to replace"""
        nl_matches = Replacer.reg.findall(nl)
        cmd_matches = Replacer.reg.findall(cmd)
        # Go through and find variable assignments
        var_to_val_map = {}

        def parse_assignment(no_brackets):
            var_and_val = no_brackets.split("=")
            if len(var_and_val) != 2:
                raise ReplacementError("Only one equal sign expected. Got", match)
            var_name_, val_ = var_and_val
            var_name_.strip()
            val_.strip()
            return var_name_, val_

        for match in nl_matches + cmd_matches:
            no_brackets = match[3:-3]
            if "=" in no_brackets:
                var_name, val = parse_assignment(no_brackets)
                # an assignment
                if not var_name.isalnum() or not var_name.lower() == var_name:
                    raise ReplacementError(
                        "Assignment name should be lowercase alphanumeric. Got", var_name)
                if var_name in var_to_val_map:
                    raise ReplacementError("Duplicate assignment in: ", nl, ", ", cmd)
                var_to_val_map[var_name] = val
        # replace
        new_nl, new_cmd = nl, cmd
        for match in nl_matches:
            # A match will have be surrounded with dash-brackets
            no_brackets = match[3:-3]
            val = no_brackets
            var_name = None
            if "=" in no_brackets:
                var_name, val = parse_assignment(no_brackets)
            elif no_brackets[0] == "$":
                var_name = no_brackets[1:]
                if var_name not in var_to_val_map:
                    raise ReplacementError("Use of unassigned value : ", no_brackets,
                                           " cmd ", cmd, " nl ", nl)
                val = var_to_val_map[var_name]

            val_words = val.split(" ")
            match_typename = val_words[0]

            # sample
            if match_typename not in self.nameToTypes:
                raise ValueError("unrecognized replacement type", match_typename,
                                 "accepted = ", self.nameToTypes)
            nlreplace, cmdreplace = self.nameToTypes[match_typename].sample_replacement(val_words)
            new_nl = new_nl.replace(match, nlreplace)
            new_cmd = new_cmd.replace(match, cmdreplace)
            if var_name:
                new_nl = new_nl.replace("[-[$"+var_name+"]-]", nlreplace)
                new_cmd = new_cmd.replace("[-[$"+var_name+"]-]", cmdreplace)

        return new_nl, new_cmd


class Replacement:
    """An individual x, y value replacement."""
    def __init__(self, nl_value: str, cmd_value: str, weight: float):
        self._nl_value = nl_value
        self._cmd_value = cmd_value
        self.weight = weight

    def get_replacement(self):
        return self._nl_value, self._cmd_value

    @classmethod
    def from_tsv(cls, file_name) -> List['Replacement']:
        """Returns a list a of replacements loaded from a tsv"""
        with open(file_name) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            return [cls(nl, cmd, int(weight)) for nl, cmd, weight in reader]


class ReplacementGroup:
    """A set of replacements of the same type to use"""
    def __init__(self, name: str, replacements: List[Replacement]):
        self.name = name
        self.replacements = replacements
        weights = [r.weight for r in replacements]
        self._sampler = WeightedRandomChooser(replacements, weights)

    def sample_replacement(self, argwords = []):
        return self._sampler.sample().get_replacement()

