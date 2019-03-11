"""Classes for doing inference with model. For now these user facing classes
are just imported right into the user program. However, in the future we will
let the Kernel be started indepently and then communicate with it via some
form of IPC so that it can be independent of the user program platform and
so multiple 'shells' can use the same model"""
from typing import Tuple

import torch

from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_common.parsing.parse_primitives import AInixParseError
from ainix_common.parsing.stringparser import AstUnparser, UnparseResult
from ainix_kernel.models.EncoderDecoder.encdecmodel import EncDecModel
from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_kernel.models.model_types import TypeTranslatePredictMetadata, ModelException
from ainix_kernel.util.serialization import restore
from ainix_common.parsing.model_specific.tokenizers import NonLetterTokenizer
import attr


@attr.s(auto_attribs=True, frozen=True)
class PredictReturn:
    success: bool
    ast: ObjectChoiceNode = None
    unparse: UnparseResult = None
    # TODO the shell interface should only have require the user code to depend
    # on ainix_common. Figure out how to move this.
    metad: TypeTranslatePredictMetadata = None
    error_message: str = None


class Interface():
    def __init__(self, file_name):
        #self.type_context, self.model, self.example_store = restore(file_name)

        # hacks
        from ainix_kernel.training import fullret_try
        model, index, replacers, type_context, loader = fullret_try.train_the_thing()
        self.type_context, self.model, self.example_store = type_context, model, index

        self.unparser = AstUnparser(self.type_context, self.model.get_string_tokenizer())

    def predict(self, utterance: str, ytype: str) -> PredictReturn:
        try:
            result, metad = self.model.predict(utterance, ytype, False)
            assert result.is_frozen
            unparse = self.unparser.to_string(result, utterance)
            return PredictReturn(
                success=True,
                ast=result,
                unparse=unparse,
                metad=metad,
                error_message=None
            )
        except ModelException as e:
            return PredictReturn(
                False,
                None,
                None,
                None, str(e)
            )
