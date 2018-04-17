import cmd_parse
import torchtext
from torchtext.data import Pipeline
from torchtext.data.dataset import Dataset
import six
from utils import WeightedRandomChooser
import re
import pudb
import csv
import constants
from collections import Counter, OrderedDict

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
            otherArgs = None
            if len(vals) > 1:
                try:
                    int(vals[0])
                    matchtypename = vals[1]
                    if len(vals) > 2:
                        otherArgs = vals[2:]
                except ValueError:
                    # if the first one is not an int, then the type must come first
                    matchtypename = vals[0]
                    if len(vals) > 1:
                        otherArgs = vals[2:]
            else:
                matchtypename = vals[0]
                
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

    @classmethod
    def from_tsv(cls, file_name):
        """Returns a list a of replacements loaded from a tsv"""
        with open(file_name) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            return [cls(nl, cmd, int(weight)) for nl, cmd, weight in reader]

class NLExample():
    def __init__(self, sequence, field):
        self.raw_sequence = sequence
        # Add in copy tokens
        self.subsequence_to_copy = {}
        self.copy_to_sequence = {}
        newTok = []
        onCopyIndex = 0
        for e in sequence:
            shouldHaveCopy = True
            shouldHaveCopy = shouldHaveCopy and (e in (constants.SOS, constants.EOS))
            shouldHaveCopy = shouldHaveCopy or (e == constants.UNK or field.vocab.freqs[e] < 3)
            shouldHaveCopy = shouldHaveCopy and (onCopyIndex < len(constants.COPY_TOKENS))
            if shouldHaveCopy:
                newCopyVal = (e,)
                if newCopyVal in self.subsequence_to_copy:
                    newTok.append(self.subsequence_to_copy[(e,)])
                else:
                    useCopyTok = constants.COPY_TOKENS[onCopyIndex]
                    newTok.append(useCopyTok)
                    self.subsequence_to_copy[newCopyVal] = useCopyTok
                    self.copy_to_sequence[useCopyTok] = newCopyVal
                    onCopyIndex += 1
            newTok.append(e)
        self.mod_text = newTok

class NLField(torchtext.data.Field):
    """A torch text field field for the NL/hybrid utterenece input commands"""
    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.
        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and
                not isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)


        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device, train):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(NLExample)): A list of object from a batch of examples.
        """
        asexamples = [NLExample(e, self) for e in batch]
        vals = [e.mod_text for e in asexamples]
        padded = self.pad(vals)
        tensor = self.numericalize(padded, device=device, train=train)
        return tensor, asexamples
    
    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        specials += constants.COPY_TOKENS
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

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
