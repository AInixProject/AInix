import cmd_parse
import torchtext
from torchtext.data import Pipeline
import six
from utils import WeightedRandomChooser
import re
import pudb

class Replacer():
    pattern = r"\[-\[.+?\]-\]"
    reg = re.compile(pattern)
    def __init__(self, types):
        self.types = types
        self.nameToTypes = {t.name: t for t in types}

    def strings_replace(self, nl, cmd):
        nlmatches = Replacer.reg.findall(nl)
        cmdmatches = Replacer.reg.findall(cmd)
        # check that has a correspondence
        nlset = set(nlmatches)
        cmdset = set(cmdmatches)
        intersection = nlset & cmdset
        # replace
        newnl, newcmd = nl, cmd
        for match in nlmatches:
            # A match will have be surrounded with dash-brackets with .'s seperating args
            vals = match[3:-3].split(".")
            if len(vals) > 1:
                try:
                    int(vals[0])
                    matchtypename = vals[1]
                except ValueError:
                    # if the first one is not an int, then the type must come first
                    matchtypename = vals[0]
                if len(vals) > 2:
                    otherArgs = vals[2:]
            else:
                matchtypename = vals[0]
                otherArgs = None
                
            # sample 
            if matchtypename not in self.nameToTypes:
                raise ValueError("unrecognized replacement type", matchtypename, "accepted = ", self.nameToTypes)
            if otherArgs:
                nlreplace, cmdreplace = self.nameToTypes[matchtypename].sample_replacement(*otherArgs)
            else:
                nlreplace, cmdreplace = self.nameToTypes[matchtypename].sample_replacement()
            newnl = newnl.replace(match, nlreplace)
            newcmd = newcmd.replace(match, cmdreplace)
        return newnl, newcmd

class ReplacementGroup():
    def __init__(self, name, replacements):
        self.name = name
        self.replacements = replacements
        weights = [r.weight for r in replacements]
        self._sampler = WeightedRandomChooser(replacements, weights)

    def sample_replacement(self, *args):
        return self._sampler.sample().get_replacement()

class Replacement():
    def __init__(self, nl_value, cmd_value, weight):
        self._nl_value = nl_value
        self._cmd_value = cmd_value
        self.weight = weight

    def get_replacement(self):
        return self._nl_value, self._cmd_value

class CommandField(torchtext.data.RawField):
    """A torch text field specifically for commands. Will parse
    the raw string commands into an AST and label programs into
    catagories"""
    def __init__(self, descriptions):
        self.descriptions = descriptions
        self.cmd_parser = cmd_parse.CmdParser(descriptions)

    def preprocess(self, x):
        # Use code from torchtext Textfield to make sure unicode
        if (six.PY2 and isinstance(x, six.string_types) and not
               isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        # send through the parser
        x = Pipeline(lambda s: self.cmd_parser.parse(s))(x)
        return x

    def process(self, batch, *args, **kargs):
        return batch
