import torch
from unittest.mock import MagicMock
import unittest.mock

from ainix_common.parsing.stringparser import StringParser
from ainix_common.tests.toy_contexts import get_toy_strings_context
from ainix_kernel.models.EncoderDecoder.decoders import TreeDecoder, TreeRNNDecoder, TreeRNNCell


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

    latents, yinds = decoder.get_latent_select_states(
        torch.Tensor(1, 4), torch.Tensor(1, 3, 4), ast)

    assert len(latents) == len(yinds) == 3
    assert latents == [out_v for _ in range(3)]
    assert yinds == [0, 2, 4]
