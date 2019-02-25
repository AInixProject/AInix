from ainix_kernel.model_util import vocab
from ainix_kernel.models.EncoderDecoder.encoders import *
from ainix_kernel.models.EncoderDecoder.encdecmodel import get_default_tokenizers, \
    make_default_query_encoder
from ainix_kernel.tests.testutils.torch_test_utils import torch_train_tester, \
    eps_eq_at, torch_epsilon_eq
from ainix_common.parsing.model_specific import parse_constants
import random
import pytest


@pytest.mark.parametrize("batch_size", (1, 2))
def test_default_encoder(batch_size):
    torch.manual_seed(0)
    random.seed(0)
    (x_tokenizer, x_vocab), _ = get_default_tokenizers()
    if x_vocab is None:
        vocab_builder = vocab.CounterVocabBuilder()
        vocab_builder.add_sequence(["a", "b", "c", "d"] + parse_constants.ALL_SPECIALS)
        x_vocab = vocab_builder.produce_vocab()
    encoder = make_default_query_encoder(x_tokenizer, x_vocab, 4)
    torch_train_tester(
        model=encoder,
        data=[(("a",), torch.Tensor([0, 1, 0, 0])),
              (("b b", ), torch.Tensor([0, 0, 3, 1])),
              (("d b", ), torch.Tensor([0, 0, -3, 1])),
              (("c d", ), torch.Tensor([2, 0, -1, 0])),
              (("c a", ), torch.Tensor([0, 0, 0, -4])),
              ],
        comparer=eps_eq_at(0.15),
        y_extractor_train=lambda y: y[0],
        y_extractor_eval=lambda y: y[0],
        criterion=nn.MSELoss(),
        max_epochs=10000,
        early_stop_loss_delta=-1e-6,
        earyl_stop_patience=500,
        batch_size=batch_size,
        shuffle=True
    )
    #summary, mem = encoder(["boop otherunk"])
    ## Make sure unks get treated the same
    #assert torch_epsilon_eq(summary,
    #                        torch.Tensor([[0, 0, 0, -4]]), epsilon=1e-2)
    # make sure memory shape looks decent
    #assert mem.shape == (1, 3, 4)


def test_len_reoder():
    vals_to_reorder = (torch.Tensor([[1,1,2,2], [3,4,4,3], [5,6,7,8]]), torch.randn(3, 4, 3))
    sorted_lens, sorting_inds, vals_after_reorder = reorder_based_off_len(
        input_lens=torch.LongTensor([2, 4, 1]),
        vals_to_reorder=vals_to_reorder
    )
    assert torch_epsilon_eq(sorted_lens, torch.LongTensor([4, 2, 1]))
    assert torch_epsilon_eq(sorting_inds, torch.LongTensor([1, 0, 2]))
    assert torch_epsilon_eq(vals_after_reorder[0], torch.Tensor([[3,4,4,3], [1,1,2,2], [5,6,7,8]]))
    put_back_together = undo_len_ordering(sorting_inds, vals_after_reorder)
    assert torch_epsilon_eq(put_back_together[0], vals_to_reorder[0])
    assert torch_epsilon_eq(put_back_together[1], vals_to_reorder[1])

