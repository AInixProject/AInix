from ainix_kernel.model_util import vocab
from ainix_kernel.models.EncoderDecoder.encoders import *
from ainix_kernel.models.EncoderDecoder.encdecmodel import _get_default_tokenizers
from ainix_kernel.tests.testutils.torch_test_utils import torch_train_tester, eps_eq_at


def test_default_encoder():
    torch.manual_seed(0)
    x_tokenizer, _ = _get_default_tokenizers()
    vocab_builder = vocab.CounterVocabBuilder()
    vocab_builder.add_sequence(["foo", "bar", "baz", "boop"])
    encoder = make_default_query_encoder(x_tokenizer, vocab_builder.produce_vocab(), 4)
    torch_train_tester(
        model=encoder,
        data=[((["foo"],), torch.Tensor([[0, 1, 0, 0]])),
              ((["foo bar"], ), torch.Tensor([[0, 0, 3, 1]])),
              ((["bar foo"], ), torch.Tensor([[0, 0, -3, 1]])),
              ((["boop baz"], ), torch.Tensor([[2, 0, -1, 0]])),
              ],
        comparer=eps_eq_at(1e-2),
        y_extractor_train=lambda y: y[0],
        y_extractor_eval=lambda y: y[0],
        criterion=nn.MSELoss(),
        max_epochs=5000,
        early_stop_loss_delta=-1e-6
    )
