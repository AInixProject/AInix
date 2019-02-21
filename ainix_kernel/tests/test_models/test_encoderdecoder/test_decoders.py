import torch
from unittest.mock import MagicMock
import unittest.mock

from ainix_common.parsing.model_specific import parse_constants
from ainix_common.parsing.model_specific.tokenizers import ModifiedStringToken, CasingModifier, \
    WhitespaceModifier
from ainix_common.parsing.stringparser import StringParser
from ainix_common.tests.toy_contexts import get_toy_strings_context
from ainix_kernel.models.EncoderDecoder.decoders import TreeDecoder, TreeRNNDecoder, TreeRNNCell, \
    get_valid_for_copy_mask
from ainix_kernel.tests.testutils.torch_test_utils import torch_epsilon_eq
from ainix_common.parsing.model_specific.parse_constants import PAD, SOS, EOS


def test_get_latents():
    out_v = torch.Tensor([[1,2,3,4]])
    mock_cell = MagicMock(return_value=(out_v, torch.Tensor(1, 4)))
    mock_selector = MagicMock()
    mock_vectorizer = MagicMock()
    mock_vocab = MagicMock()
    decoder = TreeRNNDecoder(mock_cell, mock_selector, mock_vectorizer, mock_vocab)
    tc = get_toy_strings_context()
    parser = StringParser(tc)
    ast = parser.create_parse_tree("TWO foo bar", "ToySimpleStrs")

    latents = decoder.get_latent_select_states(
        torch.Tensor(1, 4), torch.Tensor(1, 3, 4), MagicMock(), ast)

    assert len(latents) == 3
    assert latents == [out_v for _ in range(3)]


def _ms(string: str):
    return ModifiedStringToken(string, CasingModifier.CASELESS,
                               WhitespaceModifier.AFTER_SPACE_OR_SOS)


def test_get_valid_for_copy_mask():
    assert torch_epsilon_eq(
        get_valid_for_copy_mask([
            [_ms(SOS), _ms("a"), _ms(parse_constants.SPACE), _ms("b"), _ms(EOS)],
            [_ms(SOS), _ms("a"), _ms(PAD), _ms(PAD), _ms(EOS)],
        ]),
        torch.tensor([[0, 1, 0, 1, 0],
                      [0, 1, 0, 0, 0]])
    )
