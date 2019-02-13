"""
Generic testing for language modeling
"""
import math

import torch
from typing import List

from ainix_common.parsing.model_specific import tokenizers, parse_constants
from ainix_common.parsing.model_specific.tokenizers import CasingModifier, WhitespaceModifier
from ainix_kernel.model_util.lm_task_processor.lm_set_process import CookieMonsterBatchIterator, \
    CookieMonsterDataset, LMExampleTorched, LMBatch
from ainix_kernel.model_util.vocab import BasicVocab
from ainix_kernel.models.LM.cookiemonster import make_default_cookie_monster
import pytest


def make_fake_lm_example(token_inds: List[int], is_seq: bool):
    return LMExampleTorched(
        tokens=torch.LongTensor(token_inds),
        token_case_mod=torch.LongTensor([CasingModifier.CASELESS.value for _ in token_inds]),
        token_whitespace_mod=torch.LongTensor([WhitespaceModifier.AFTER_SPACE_OR_SOS.value
                                               for _ in token_inds]),
        is_sequential=is_seq,
        mask_inds=torch.LongTensor([]),
        mask_expected_ind=torch.LongTensor([]),
        mask_expected_case=torch.LongTensor([])
    )


def test_basic_e2e():
    vocab = BasicVocab(["a", "b", "c", "d", "e", "f", "g", "h", "i"] + parse_constants.ALL_SPECIALS)
    model = make_default_cookie_monster(vocab, hidden_size_base=16)
    pad_ind = vocab.token_to_index(parse_constants.PAD)
    split_ind = vocab.token_to_index(parse_constants.TASK_SPLITTER)
    examples = [
        make_fake_lm_example([1, 2, 3, split_ind, 1, 2], True),
        make_fake_lm_example([1, 2, 3, split_ind, 6, 4], False),
        make_fake_lm_example([1, 3, 2, split_ind, 3, 2], True),
        make_fake_lm_example([6, 5, 4, split_ind, 6, 5], True),
        make_fake_lm_example([1, 2, 3, split_ind, 5, 6], False),
        make_fake_lm_example([6, 5, 3, split_ind, 1, 1], False),
    ]
    model.start_train_session()
    for i in range(400):
        loss = model.train_batch(LMBatch.from_example_list(examples, pad_ind))
        if loss < math.log(1.25):
            break
    else:
        pytest.fail("Did not converge in expected number of iterations")
