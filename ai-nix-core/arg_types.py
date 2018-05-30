import constants
import tokenizers
import pudb
class ArgumentType():
    def __init__(self):
        self.model_data = None
        self.requires_value = False

class StoreTrue(ArgumentType):
    def as_shell_string(self, value):
        return ''

class Stringlike(ArgumentType):
    def __init__(self):
        super(Stringlike, self).__init__()
        self.requires_value = True
        self.is_multi_word = True

    def parse_value(self, value, run_context, copyfromexample, is_eval = False):
        """Takes in a value and converts it to friendly representation"""
        preproc = run_context.nl_field.preprocess(value)
    
        #pudb.set_trace()
        # hacky copy
        preproc_cp = copyfromexample.insert_copies(preproc)

        padded = run_context.nl_field.pad([preproc_cp])
        (tensor, lengths) = run_context.nl_field.numericalize(
                padded, 
                device = 0 if run_context.use_cuda else -1, 
                train = not is_eval
        )
        return tensor

    def train_value(self, encoding, expected_value, run_context, meta_model):
        return meta_model.std_decode_train(encoding, run_context, expected_value)

    def eval_value(self, encoding, run_context, meta_model, copyfromexample):
        predSequence = meta_model.std_decode_eval(encoding, run_context)
        #pudb.set_trace()
        copied = []
        for p in predSequence:
            if p in constants.COPY_TOKENS:
                if p in copyfromexample.copy_to_sequence:
                    copied.extend(copyfromexample.copy_to_sequence[p])
                else:
                    copied.append(p)
            else:
                copied.append(p)

        return tokenizers.nonascii_untokenize(" ".join(copied))

    def as_shell_string(self, value):
        return value

class FileList(Stringlike):
    def __init__(self):
        super(FileList, self).__init__()
        self.is_multi_word = True

class SingleFile(Stringlike):
    def __init__(self):
        super(SingleFile, self).__init__()
        self.is_multi_word = False
