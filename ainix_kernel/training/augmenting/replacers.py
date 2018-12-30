import csv
import re
from typing import List

from ainix_kernel.util.sampling import WeightedRandomChooser


class ReplacementError(ValueError):
    pass


class Replacer:
    pattern = r"\[-\[.+?\]-\]"
    reg = re.compile(pattern)

    def __init__(self, types):
        self.types = types
        self.nameToTypes = {t.name: t for t in types}

    def strings_replace(self, nl, cmd):
        nl_matches = Replacer.reg.findall(nl)
        cmd_matches = Replacer.reg.findall(cmd)
        # Go through and find variable assignments
        var_to_val_map = {}

        def parse_assignment(nobrackets):
            var_and_val = nobrackets.split("=")
            if len(var_and_val) != 2:
                raise ReplacementError("Only one equal sign expected. Got", match)
            var_name, val = var_and_val
            var_name.strip()
            val.strip()
            return var_name, val

        for match in nl_matches + cmd_matches:
            nobrackets = match[3:-3]
            replaceExpression = None
            if "=" in nobrackets:
                varname, val = parse_assignment(nobrackets)
                # an assignment
                if not varname.isalnum() or not varname.lower() == varname:
                    raise ReplacementError(
                        "Assignment name should be lowercase alphanumeric. Got", varname)
                if varname in var_to_val_map:
                    raise ReplacementError("Duplicate assignment in: ", nl, ", ", cmd)
                var_to_val_map[varname] = val
        # replace
        newnl, newcmd = nl, cmd
        for match in nl_matches:
            # A match will have be surrounded with dash-brackets
            nobrackets = match[3:-3]
            val = nobrackets
            varname = None
            if "=" in nobrackets:
                varname, val = parse_assignment(nobrackets)
            elif nobrackets[0] == "$":
                varname = nobrackets[1:]
                if varname not in var_to_val_map:
                    raise ReplacementError("Use of unassigned value : ", nobrackets, " cmd ", cmd, " nl ", nl)
                val = var_to_val_map[varname]

            valWords = val.split(" ")
            matchtypename = valWords[0]

            # sample
            if matchtypename not in self.nameToTypes:
                raise ValueError("unrecognized replacement type", matchtypename, "accepted = ", self.nameToTypes)
            nlreplace, cmdreplace = self.nameToTypes[matchtypename].sample_replacement(valWords)
            newnl = newnl.replace(match, nlreplace)
            newcmd = newcmd.replace(match, cmdreplace)
            if varname:
                newnl = newnl.replace("[-[$"+varname+"]-]", nlreplace)
                newcmd = newcmd.replace("[-[$"+varname+"]-]", cmdreplace)

        return newnl, newcmd


class Replacement:
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
    def __init__(self, name: str, replacements: List[Replacement]):
        self.name = name
        self.replacements = replacements
        weights = [r.weight for r in replacements]
        self._sampler = WeightedRandomChooser(replacements, weights)

    def sample_replacement(self, argwords = []):
        return self._sampler.sample().get_replacement()

