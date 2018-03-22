import cmd_parse
import torchtext
from torchtext.data import Pipeline
import six

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
