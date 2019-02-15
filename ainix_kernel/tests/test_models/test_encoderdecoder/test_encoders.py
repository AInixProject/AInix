from ainix_kernel.model_util import vocab
from ainix_kernel.models.EncoderDecoder.encoders import *
from ainix_kernel.models.EncoderDecoder.encdecmodel import _get_default_tokenizers, \
    make_default_query_encoder
from ainix_kernel.tests.testutils.torch_test_utils import torch_train_tester, \
    eps_eq_at, torch_epsilon_eq
from ainix_common.parsing.model_specific import parse_constants
import random


def test_default_encoder():
    torch.manual_seed(0)
    x_tokenizer, _ = _get_default_tokenizers()
    vocab_builder = vocab.CounterVocabBuilder()
    vocab_builder.add_sequence(["foo", "bar", "baz", "boop"] + parse_constants.ALL_SPECIALS)
    encoder = make_default_query_encoder(x_tokenizer, vocab_builder.produce_vocab(), 4)
    torch_train_tester(
        model=encoder,
        data=[(("foo",), torch.Tensor([0, 1, 0, 0])),
              (("foo bar", ), torch.Tensor([0, 0, 3, 1])),
              (("bar foo", ), torch.Tensor([0, 0, -3, 1])),
              (("boop baz", ), torch.Tensor([2, 0, -1, 0])),
              (("boop imunk", ), torch.Tensor([0, 0, 0, -4])),
              ],
        comparer=eps_eq_at(1e-2),
        y_extractor_train=lambda y: y[0],
        y_extractor_eval=lambda y: y[0],
        criterion=nn.MSELoss(),
        max_epochs=5000,
        early_stop_loss_delta=-1e-6
    )
    #summary, mem = encoder(["boop otherunk"])
    ## Make sure unks get treated the same
    #assert torch_epsilon_eq(summary,
    #                        torch.Tensor([[0, 0, 0, -4]]), epsilon=1e-2)
    # make sure memory shape looks decent
    #assert mem.shape == (1, 3, 4)


def test_default_encoder_batched():
    torch.manual_seed(0)
    random.seed(0)
    x_tokenizer, _ = _get_default_tokenizers()
    vocab_builder = vocab.CounterVocabBuilder()
    vocab_builder.add_sequence(["foo", "bar", "baz", "boop"] + parse_constants.ALL_SPECIALS)
    encoder = make_default_query_encoder(x_tokenizer, vocab_builder.produce_vocab(), 4)
    torch_train_tester(
        model=encoder,
        data=[(("foo",), torch.Tensor([0, 1, 0, 0])),
              (("foo bar", ), torch.Tensor([0, 0, 3, 1])),
              (("bar foo", ), torch.Tensor([0, 0, -3, 1])),
              (("boop baz", ), torch.Tensor([2, 0, -1, 0])),
              (("boop imunk", ), torch.Tensor([0, 0, 0, -4]))
              ],
        comparer=eps_eq_at(1e-2),
        y_extractor_train=lambda y: y[0],
        y_extractor_eval=lambda y: y[0],
        criterion=nn.MSELoss(),
        max_epochs=5000,
        early_stop_loss_delta=-1e-6,
        earyl_stop_patience=100,
        batch_size=2,
        shuffle=True
    )
    summary, mem = encoder(["boop otherunk"])
    # Make sure unks get treated the same
    assert torch_epsilon_eq(summary,
                            torch.Tensor([[0, 0, 0, -4]]), epsilon=1e-2)
    # make sure memory shape looks decent
    assert mem.shape == (1, 3, 4)


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

