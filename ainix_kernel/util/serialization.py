from typing import Tuple

import torch

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.models.EncoderDecoder.encdecmodel import EncDecModel
from ainix_kernel.models.model_types import StringTypeTranslateCF
from ainix_kernel.specialtypes import allspecials


def serialize(model: StringTypeTranslateCF, loader: TypeContextDataLoader, save_path: str):
    ser = {
        "model": model.get_save_state_dict(),
        "type_loader": loader.get_save_state_dict()
    }
    torch.save(ser, save_path)


def restore(file_name) -> Tuple[TypeContext, StringTypeTranslateCF]:
    save_dict = torch.load(file_name)
    type_context, loader = TypeContextDataLoader.restore_from_save_dict(save_dict['type_loader'])
    allspecials.load_all_special_types(type_context)
    type_context.fill_default_parsers()
    # Hard code model type. Should not do this...
    model = EncDecModel.create_from_save_state_dict(save_dict['model'], type_context, None)
    model.end_train_session()
    return type_context, model
