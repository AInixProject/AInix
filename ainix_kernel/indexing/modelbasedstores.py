import torch

from ainix_common.parsing.ast_components import AstObjectChoiceSet
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import BasicExampleStore
import numpy as np


class CacheingExampleStore(BasicExampleStore):
    def __init__(self, type_context: TypeContext):
        super().__init__(type_context)

    def store_example_info(
        self,
        x_val_id: int,
        x_val_with_replacements: str,
        y_ast_set: AstObjectChoiceSet,
        summary: torch.Tensor,
        tokens: np.ndarray,
        encoded_tokens: torch.Tensor
    ):
        pass


