from typing import Tuple

import torch

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import ExamplesStore
from ainix_kernel.models.EncoderDecoder.encdecmodel import EncDecModel
from ainix_kernel.models.model_types import StringTypeTranslateCF
from ainix_kernel.specialtypes import allspecials
from ainix_kernel.training.evaluate import EvaluateLogger
from ainix_kernel.training.train_contexts import load_all_examples


def serialize(
    model: StringTypeTranslateCF,
    loader: TypeContextDataLoader,
    save_path: str,
    eval_results: EvaluateLogger = None,
    trained_epochs = None
):
    ser = {
        "version": 0,
        "model": model.get_save_state_dict(),
        "type_loader": loader.get_save_state_dict(),
        "eval_results": eval_results,
        "trained_epochs": trained_epochs
    }
    torch.save(ser, save_path)


def restore(file_name) -> Tuple[TypeContext, StringTypeTranslateCF, ExamplesStore]:
    save_dict = torch.load(file_name)
    type_context, loader = TypeContextDataLoader.restore_from_save_dict(save_dict['type_loader'])
    allspecials.load_all_special_types(type_context)
    type_context.finalize_data()
    # Hard code model type. Should not do this...
    need_example_store = save_dict['model']['need_example_store']
    if need_example_store:
        # TODO (DNGros) smart restoring.
        example_store = load_all_examples(type_context)
    else:
        example_store = None
    model = EncDecModel.create_from_save_state_dict(save_dict['model'], type_context, example_store)
    model.end_train_session()
    return type_context, model, example_store
