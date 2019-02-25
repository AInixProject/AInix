"""A proof of concept hacky exploration of how well just template nearest
match might work"""
import random
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ainix_common.parsing.loader import TypeContextDataLoader
from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_common.parsing.model_specific.tokenizers import get_default_pieced_tokenizer_word_list, \
    AstValTokenizer
from ainix_common.parsing.typecontext import TypeContext
from ainix_kernel.indexing.examplestore import DataSplits
from ainix_kernel.model_util.operations import avg_pool
from ainix_kernel.model_util.vocab import Vocab, BasicVocab
from ainix_kernel.models.LM.cookiemonster import CookieMonsterBaseEncoder, \
    PretrainPoweredQueryEncoder, make_default_cookie_monster_base
from ainix_kernel.specialtypes import allspecials
from ainix_kernel.training.augmenting.replacers import get_all_replacers
from ainix_kernel.training.train_contexts import ALL_EXAMPLE_NAMES, load_all_examples


class PretrainAvger(PretrainPoweredQueryEncoder):
    def _sumarize(self, hidden, input_lens):
        hidden = avg_pool(hidden, input_lens)
        return hidden


def _get_default_tokenizers() -> Tuple[
    Tuple[tokenizers.Tokenizer, Optional[Vocab]],
    tokenizers.Tokenizer
]:
    # Copy pasta from encdecmodel.py. bad. fix
    """Returns tuple (default x tokenizer, default y tokenizer)"""
    word_piece_tok, word_list = get_default_pieced_tokenizer_word_list()
    x_vocab = BasicVocab(word_list + parse_constants.ALL_SPECIALS)
    return (word_piece_tok, x_vocab), AstValTokenizer()


if __name__ == "__main__":
    pretrained_checkpoint_path = "../../checkpoints/" \
                                 "lmchkp_iter152k_200_2rnn_total3.29_ns0.47_lm2.82.pt"
    output_size = 200
    (x_tokenizer, query_vocab), y_tokenizer = _get_default_tokenizers()
    base_enc = make_default_cookie_monster_base(
        query_vocab, output_size)
    model = PretrainPoweredQueryEncoder.create_with_pretrained_checkpoint(
        pretrained_checkpoint_path,
        x_tokenizer, query_vocab, output_size, freeze_base=True
    )
    model.eval()


    type_context = TypeContext()
    loader = TypeContextDataLoader(type_context, up_search_limit=4)
    loader.load_path("builtin_types/generic_parsers.ainix.yaml")
    loader.load_path("builtin_types/command.ainix.yaml")
    loader.load_path("builtin_types/paths.ainix.yaml")
    allspecials.load_all_special_types(type_context)

    for f in ALL_EXAMPLE_NAMES:
        loader.load_path(f"builtin_types/{f}.ainix.yaml")
    type_context.finalize_data()

    index = load_all_examples(type_context)
    #index = load_tellia_examples(type_context)

    print("num docs", index.backend.index.doc_count())
    replacers = get_all_replacers()


    train_splits = (DataSplits.TRAIN,)
    all_ex_list = list(index.get_all_examples(train_splits))
    random.shuffle(all_ex_list)
    processed_x_raws = set()
    summaries = []
    examples = []
    for example in tqdm(all_ex_list):
        if example.xquery in processed_x_raws:
            continue
        rxs = list()
        for i in range(20):
            x, y = replacers.strings_replace(example.xquery, example.ytext)
            rxs.append(x)
        if set(rxs) == 0:
            # If all the same (no replacers), just take the set version
            rxs = set(rxs)
        summary, memory, tokens = model(list(rxs))
        #summary, memory, tokens = model([example.xquery])
        summaries.append(torch.mean(summary, dim=0))
        examples.append((example.xquery, example.ytext))
        processed_x_raws.add(example.xquery)
    summaries = torch.stack(summaries)
    print(summaries.shape)

    while True:
        q = input("Query: ")
        summary, memory, tokens = model([q])
        sims = F.cosine_similarity(summary, summaries)
        print(sims)
        top_sims, top_inds = torch.topk(sims, 5)
        for score, ind in zip(top_sims, top_inds):
            print(score)
            print(examples[int(ind)])
