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


def make_fake_lm_example(token_inds: List[int], is_seq: bool, device: torch.device):
    return LMExampleTorched(
        tokens=torch.tensor(token_inds, device=device),
        token_case_mod=torch.tensor([CasingModifier.CASELESS.value for _ in token_inds],
                                    device=device),
        token_whitespace_mod=torch.tensor([WhitespaceModifier.AFTER_SPACE_OR_SOS.value
                                               for _ in token_inds],
                                          device=device),
        is_sequential=is_seq,
        mask_inds=torch.tensor([], device=device),
        mask_expected_ind=torch.tensor([], device=device),
        mask_expected_case=torch.tensor([], device=device)
    )


@pytest.mark.parametrize("use_cuda", (False, True))
def test_basic_e2e(use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("Cuda not available")
    vocab = BasicVocab(["a", "b", "c", "d", "e", "f", "g", "h", "i"] + parse_constants.ALL_SPECIALS)
    model = make_default_cookie_monster(vocab, hidden_size_base=16, use_cuda=use_cuda)
    pad_ind = vocab.token_to_index(parse_constants.PAD)
    split_ind = vocab.token_to_index(parse_constants.TASK_SPLITTER)
    dvc = torch.device("cuda" if use_cuda else "cpu")
    examples = [
        make_fake_lm_example([1, 2, 3, split_ind, 1, 2], True, dvc),
        make_fake_lm_example([1, 2, 3, split_ind, 6, 4], False, dvc),
        make_fake_lm_example([1, 3, 2, split_ind, 3, 2], True, dvc),
        make_fake_lm_example([6, 5, 4, split_ind, 6, 5], True, dvc),
        make_fake_lm_example([1, 2, 3, split_ind, 5, 6], False, dvc),
        make_fake_lm_example([6, 5, 3, split_ind, 1, 1], False, dvc),
    ]
    model.start_train_session()
    for i in range(400):
        ns_loss, lm_loss, total_loss = model.train_batch(
            LMBatch.from_example_list(examples, pad_ind, dvc))
        if ns_loss < math.log(1.25) and lm_loss < 0.0001 and total_loss < math.log(1.25):
            break
    else:
        pytest.fail("Did not converge in expected number of iterations")


# TODO (DNGros): Add test for mask task