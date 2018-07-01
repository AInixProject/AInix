import pudb
import pickle
from ainix_kernel.custom_fields import NLExample
from ainix_kernel import serialize_tools
import torch

class Bunch:
    """http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/"""
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class Interface():
    def __init__(self, fn = "../ainix_kernel/savedtest.pkl"):
        self.model = serialize_tools.restore(fn)
        self.nl_field = self.model.run_context.nl_field
        self.run_context = self.model.run_context

    def predict(self, utterance):
        # NOTE (DNGros): when upgrade torch text will likely need to pass in actual
        # device instead of -1. Also, this wont work if cuda.
        prepoc = self.nl_field.preprocess(utterance)
        processed = self.nl_field.process([prepoc], -1, train=False)
        fake_batch = Bunch(nl = processed, command = None)
        pred, _, _ = self.model.eval_step(None, fake_batch)
        return pred[0].as_shell_string()
        
