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
import tokenizers

class ReplacementError(ValueError):
    pass

class Replacer():
    pattern = r"\[-\[.+?\]-\]"
    reg = re.compile(pattern)
    def __init__(self, types):
        self.types = types
        self.nameToTypes = {t.name: t for t in types}

    def strings_replace(self, nl, cmd):
        nlmatches = Replacer.reg.findall(nl)
        cmdmatches = Replacer.reg.findall(cmd)
        # Go through and find variable assignments
        varToValMap = {}
        def parseAsignment(nobrackets):
            varAndVal = nobrackets.split("=")
            if len(varAndVal) != 2:
                raise ReplacementError("Only one equal sign expected. Got", match)
            varname, val = varAndVal
            varname.strip()
            val.strip()
            return varname, val

        for match in nlmatches + cmdmatches:
            nobrackets = match[3:-3]
            replaceExpression = None
            if "=" in nobrackets:
                varname, val = parseAsignment(nobrackets)
                # an assignment
                if not varname.isalnum() or not varname.lower() == varname:
                    raise ReplacementError("Assignment name should be lowercase alphanumeric. Got", varname)
                if varname in varToValMap:
                    raise ReplacementError("Duplicate assignment in: ", nl, ", ", cmd)
                varToValMap[varname] = val
        # replace
        newnl, newcmd = nl, cmd
        for match in nlmatches:
            # A match will have be surrounded with dash-brackets
            nobrackets = match[3:-3]
            val = nobrackets
            varname = None
            if "=" in nobrackets:
                varname, val = parseAsignment(nobrackets)
            elif nobrackets[0] == "$":
                varname = nobrackets[1:]
                if varname not in varToValMap:
                    raise ReplacementError("Use of unassigned value : ", nobrackets, " cmd ", cmd, " nl ", nl)
                val = varToValMap[varname]

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

class ReplacementGroup():
    def __init__(self, name, replacements):
        self.name = name
        self.replacements = replacements
        weights = [r.weight for r in replacements]
        self._sampler = WeightedRandomChooser(replacements, weights)

    def sample_replacement(self, argwords = []):
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
        self.subsequence_to_copy, self.copy_to_sequence, self.mod_text = \
                NLExample.create_copies(sequence, field.vocab)

    @staticmethod
    def create_copies(sequence, vocab):
        """Creates a set of relevant copy tokens from sequence and vocab"""
        subsequence_to_copy = {}
        copy_to_sequence = {}
        newTok = []
        onCopyIndex = 0

        assert constants.UNK not in sequence, "Unexpected unk in sequence?"
        split_words = tokenizers.split_tokenization(sequence)
        for word_tokens in split_words:
            combined_word = "".join(word_tokens)
            shouldHaveCopy = False
            shouldHaveCopy |= vocab.freqs[combined_word] < 3
            shouldHaveCopy &= onCopyIndex < len(constants.COPY_TOKENS)
            if shouldHaveCopy:
                newCopyVal = word_tokens
                if newCopyVal in subsequence_to_copy:
                    newTok.append(subsequence_to_copy[newCopyVal])
                else:
                    useCopyTok = constants.COPY_TOKENS[onCopyIndex]
                    newTok.append(useCopyTok)
                    subsequence_to_copy[newCopyVal] = useCopyTok
                    copy_to_sequence[useCopyTok] = newCopyVal
                    onCopyIndex += 1
            newTok.extend(word_tokens)
        return subsequence_to_copy, copy_to_sequence, newTok

    def insert_copies(self, value):
        """Takes in a tokenized value and returns copy tokens replacing the appropriate values.
        This is used during training where given an string expected argument value, all the
        proper substitutions."""
        val_with_cp = []
        i = 0
        while i < len(value):
            for sequence, copytoken in self.subsequence_to_copy.items():
                match = all([i + si < len(value) and sequence[si] == value[i + si] 
                    for si in range(len(sequence))])
                if match:
                    val_with_cp.append(copytoken)
                    i += len(sequence)
                    break
            else:
                val_with_cp.append(value[i])
                i += 1
        return val_with_cp

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
        specials += [constants.SPACE]
        specials += constants.COPY_TOKENS
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

class CommandField(torchtext.data.RawField):
    """A torch text field specifically for commands. Will parse
    the raw string commands into an AST and label programs into
    catagories"""
    def __init__(self, descriptions, use_cuda):
        self.descriptions = descriptions
        self.cmd_parser = cmd_parse.CmdParser(descriptions, use_cuda = use_cuda)

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
