import itertools
from typing import Tuple, Set, Iterable

import more_itertools
import torch

from ainix_common.parsing.ast_components import AstObjectChoiceSet, ObjectChoiceNode
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import BasicExampleStore, ExamplesStore, XValue
import numpy as np
import attr

from ainix_kernel.models.EncoderDecoder.encoders import StringQueryEncoder
from ainix_kernel.training.augmenting.replacers import ReplacementSampling


@attr.s(auto_attribs=True, frozen=True)
class EncoderCacheEntry:
    x_val_with_replacements: str
    y_ast_set: AstObjectChoiceSet
    summary: torch.Tensor
    tokens: np.ndarray
    encoded_tokens: torch.Tensor


class EncoderCache:
    def __init__(self, store_to_cache: ExamplesStore):
        self.store_to_cache = store_to_cache
        self.x_vals_cache = [[] for _ in range(store_to_cache.get_doc_count())]

    def add_entry(
        self,
        x_val_id: int,
        entry: EncoderCacheEntry
    ):
        assert not entry.summary.requires_grad
        self.x_vals_cache[x_val_id].append(entry)


def cache_examples_from_iterable(
    cache: EncoderCache,
    model: StringQueryEncoder,
    data_iterator: Iterable[Tuple[
        XValue, str, AstObjectChoiceSet, ObjectChoiceNode,
        Set[str], ReplacementSampling
    ]],
    batch_size: int = 1
):
    batches_iter = more_itertools.chunked(data_iterator, batch_size)
    # TODO (DNGros): Don't store redunant examples for things without replace
    for batch in batches_iter:
        xstrs = [data[0].x_text for data in batch]
        summaries, embeddings, tokens = model(xstrs)
        assert len(summaries) == len(embeddings) == len(tokens) == len(batch)
        for summary, embed, toks, bstuff \
            in zip(summaries, embeddings, tokens, batch):
            # TODO remove pads
            xval, replaced_x_query, y_ast_set, this_example_ast, y_texts, rsample = bstuff
            cache.add_entry(
                xval.id,
                EncoderCacheEntry(
                    x_val_with_replacements=replaced_x_query,
                    y_ast_set=y_ast_set,
                    summary=summary.detach(),
                    tokens=toks,
                    encoded_tokens=embed.detach()
                )
            )
            