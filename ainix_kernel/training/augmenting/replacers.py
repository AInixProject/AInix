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
    [-[1=NUMBER --lessthan 15]-]
which could be used to give parameters to the replacers in case the user wanted
somehow use that, but this isn't actually really used or builtout.

This code was written before the newest model architecture, and could probably
be rewritten better. Most importantly we should probably be able to make these
with .yaml files or something.
"""
import csv
import random
import re
from typing import List, Tuple, Dict, Set

from ainix_kernel.util.sampling import WeightedRandomChooser


class ReplacementError(ValueError):
    pass

_IMPLICIT_REPLACER_ARG_NAME_PREFIX = "implicitargname"


class Replacer:
    """A class that will fill in the values """
    pattern = r"\[-\[.+?\]-\]"
    reg = re.compile(pattern)

    def __init__(self, types: List['ReplacementGroup']):
        self.types = types
        self.name_to_types: Dict[Dict, 'ReplacementGroup'] = {t.name: t for t in types}

    def create_replace_sampling(self, x: str, replace_seed: int = None) -> 'ReplacementSampling':
        """Creates a replacement sampling which can be reused multiple times

        Args:
            x: the string we are replacing in.
            replace_seed: A seed to do the replacement with
        """
        random_instance = random.Random(replace_seed) if replace_seed is not None else None
        no_brackets = self.get_bracketless_matches(x)
        # Go through and find variable assignments
        var_to_val_map = {}
        for match in no_brackets:
            if "=" in match:
                var_name, val = _split_replacement_assignment(match)
            else:
                # There isn't an explicit var_name given. So just use the lower
                # case version of the type name. So for example its ok to do something
                # like [-[FILE]-] as long as only one FILE replacer in there
                if " " in match:
                    raise ReplacementError("Replacements with args must be given an explicit name")
                var_name = _IMPLICIT_REPLACER_ARG_NAME_PREFIX + match.lower()
                val = match
            if not var_name.isalnum() or not var_name.lower() == var_name or " " in var_name:
                raise ReplacementError(
                    "Assignment name should be lowercase alphanumeric. Got", var_name)
            if var_name in var_to_val_map:
                raise ReplacementError("Duplicate assignment in: ", x)
            var_to_val_map[var_name] = val
        # create actual replacements
        var_to_x_y_vals: Dict[str, Tuple[str, str]] = {}
        for var, val in var_to_val_map.items():
            val_words = val.split(" ")
            match_typename = val_words[0]
            if match_typename not in self.name_to_types:
                raise ReplacementError("unrecognized replacement type", match_typename,
                                       "accepted = ", self.name_to_types)
            repl_group = self.name_to_types[match_typename]
            x_replace, y_replace = repl_group.sample_replacement(val_words, random_instance)
            var_to_x_y_vals[var] = (x_replace, y_replace)
        return ReplacementSampling(var_to_x_y_vals)

    def strings_replace(self, x: str, y: str, replace_seed: int = None) -> Tuple[str, str]:
        """The main method used to replace"""
        sampling = self.create_replace_sampling(x, replace_seed)
        return sampling.replace_x(x), sampling.replace_y(y)

    @staticmethod
    def get_bracketless_matches(in_str):
        matches = Replacer.reg.findall(in_str)
        # A match will have be surrounded with dash-brackets. Remove those.
        no_brackets = [match[3:-3] for match in matches]
        return no_brackets

    def check_if_string_has_replacement_spots(self, string: str) -> bool:
        """Test whether a string has places that could be replaced"""
        return len(self.get_bracketless_matches(string)) > 0

def _split_replacement_assignment(no_brackets: str):
    """Separates the name and the value of a replacement item"""
    var_and_val = no_brackets.split("=")
    if len(var_and_val) != 2:
        raise ReplacementError(f"Only one equal sign expected. Got {no_brackets}")
    var_name, val = var_and_val
    var_name.strip()
    val.strip()
    return var_name, val


class ReplacementSampling:
    """This thing is returned when you run a Replacer on only a "x" value. This
    will pick concrete samplings for each replacer in that x value which can
    reapplied repeatedly to multiple "x" or "y" values filling in the same values
    in each."""
    def __init__(self, var_to_x_y_vals: Dict[str, Tuple[str, str]]):
        self.var_to_x_y_vals = var_to_x_y_vals

    def replace_x(self, in_x_str: str) -> str:
        if len(self.var_to_x_y_vals) == 0:
            return in_x_str
        vars_used = set()
        new_str, latest_vars_used = self._fill_in_var_retrieves(in_x_str, True)
        vars_used |= latest_vars_used
        new_str, latest_vars_used = self._fill_in_var_defines(new_str)
        vars_used |= latest_vars_used
        # not sure if actually want to force that all vars be used or not
        # self._verify_all_vars_used(vars_used, in_x_str)
        self._verify_no_replacers_left(new_str)
        return new_str

    def replace_y(self, in_y_str: str) -> str:
        if len(self.var_to_x_y_vals) == 0:
            self._verify_no_replacers_left(in_y_str)
            return in_y_str
        vars_used = set()
        new_str, latest_vars_used = self._fill_in_var_retrieves(in_y_str, False)
        vars_used |= latest_vars_used
        # not sure if actually want to force that all vars be used or not
        # self._verify_all_vars_used(vars_used, in_y_str)
        self._verify_no_replacers_left(new_str)
        return new_str

    def _fill_in_var_retrieves(self, in_str: str, is_x: bool) -> Tuple[str, Set[str]]:
        """Fills in any usages of a var with the actual val. This includes both
        dollar sign accesses and the just the left empty type name replacements"""
        vars_used = set()
        new_str = in_str
        for var, (x, y) in self.var_to_x_y_vals.items():
            val_to_use = x if is_x else y
            is_a_implicit_arg_name = var.startswith(_IMPLICIT_REPLACER_ARG_NAME_PREFIX)
            if is_a_implicit_arg_name:
                actual_var = var[len(_IMPLICIT_REPLACER_ARG_NAME_PREFIX):].upper()
                new_str = new_str.replace(f"[-[{actual_var}]-]", val_to_use)
            else:
                new_str = new_str.replace(f"[-[${var}]-]", val_to_use)
            if in_str != new_str:
                vars_used.add(var)
        return new_str, vars_used

    def _fill_in_var_defines(self, in_x_str: str) -> Tuple[str, Set[str]]:
        no_brackets = Replacer.get_bracketless_matches(in_x_str)
        vars_used = set()
        new_str = in_x_str
        for match in no_brackets:
            if "=" in match:
                var, val = _split_replacement_assignment(match)
                fill_str, _ = self.var_to_x_y_vals[var]
                new_str = new_str.replace(f"[-[{match}]-]", fill_str)
                vars_used.add(var)
        return new_str, vars_used

    def _verify_all_vars_used(self, used_vars: Set[str], string: str):
        if len(used_vars) != len(self.var_to_x_y_vals):
            raise ValueError(f"Not all vars used in {string}. Used {used_vars}. But "
                             f"options are {self.var_to_x_y_vals.keys()}")

    def _verify_no_replacers_left(self, string):
        if len(Replacer.get_bracketless_matches(string)) != 0:
            raise ReplacementError(f"Still replacers remaining. {string}"
                                   f"Note, if this is a Y, there may be an issue with the"
                                   f"X string (for example misplaced brackets or something)")


class Replacement:
    """An individual x, y value replacement."""
    def __init__(self, x: str, y: str, weight: float):
        self._x = x
        self._y = y
        self.weight = weight

    def get_replacement(self):
        return self._x, self._y

    @classmethod
    def from_tsv(cls, file_name) -> List['Replacement']:
        """Returns a list a of replacements loaded from a tsv"""
        with open(file_name) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            return [cls(x, y, int(weight)) for x, y, weight in reader]


class ReplacementGroup:
    """A set of replacements of a certain type to use

    Args:
        name: the name of the replacement group. This is the value that goes in
            the dash-brackets. So like in [-[FILE]-] the name is "FILE". This
            should be all caps.
    """
    def __init__(self, name: str, replacements: List[Replacement]):
        if not name.isupper():
            raise ValueError("names of replacer groups should be uppercase")
        self.name = name
        self.replacements = replacements
        weights = [r.weight for r in replacements]
        self._sampler = WeightedRandomChooser(replacements, weights)

    def sample_replacement(self, argwords = [], random_instance: random.Random = None):
        return self._sampler.sample(random_instance).get_replacement()


def seed_from_x_val(x: str, offset: int = 0) -> int:
    return (hash(x) + offset) % 100000


def get_all_replacers() -> Replacer:
    import os
    dirname, filename = os.path.split(os.path.abspath(__file__))
    repl_groups = [
        ReplacementGroup(fn, Replacement.from_tsv(f"{dirname}/data/{fn}.tsv"))
        for fn in ('FILENAME', "DIRNAME", "ENGWORD")
    ]
    replacer = Replacer(repl_groups)
    return replacer