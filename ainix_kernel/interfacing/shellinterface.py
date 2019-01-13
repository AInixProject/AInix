"""Classes for doing inference with model. For now these user facing classes
are just imported right into the user program. However, in the future we will
let the Kernel be started indepently and then communicate with it via some
form of IPC so that it can be independent of the user program platform and
so multiple 'shells' can use the same model"""
from typing import Tuple

import torch

from ainix_common.parsing.stringparser import AstUnparser
from ainix_kernel.models.EncoderDecoder.encdecmodel import EncDecModel
from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_kernel.models.model_types import TypeTranslatePredictMetadata
from ainix_kernel.util.serialization import restore
from ainix_common.parsing.model_specific.tokenizers import NonLetterTokenizer


class Interface():
    def __init__(self, file_name):
        self.type_context, self.model = restore(file_name)
        self.unparser = AstUnparser(self.type_context, self.model.get_string_tokenizer())

    def predict(self, utterance: str, ytype: str) -> Tuple[str, TypeTranslatePredictMetadata]:
        result, metad = self.model.predict(utterance, ytype, False)
        assert result.is_frozen
        unparse = self.unparser.to_string(result, utterance)
        return unparse.total_string, metad
