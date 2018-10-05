from abc import ABC, abstractmethod

import torch
from torch import nn
from typing import Iterable, Tuple, Type
from ainix_kernel.model_util.tokenizers import Tokenizer
from ainix_kernel.model_util.vectorizers import VectorizerBase
from ainix_kernel.model_util.vocab import Vocab
from ainix_kernel.model_util.rnn_encoder import EncoderRNN


class QueryEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, queries: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: An iterable representing a batch of query strings to encode

        Returns:
            A tuple of tensors. The first is the of dims (batch_size, dim_size)
            which represents a "summarization" of the entire query's meaning.
            The second is a tensor of size (batch_size, seq_len, hidden_size)
            which represents a contextualized encoding for every token in the
            input (which could be used for something like an attention or copy
            mechanism in the decoder).
        """
        raise NotImplemented()


class StringQueryEncoder(QueryEncoder):
    def __init__(
        self,
        tokenizer: Tokenizer,
        query_vocab: Vocab,
        query_vectorizer: VectorizerBase,
        internal_encoder: nn.Module
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.query_vocab = query_vocab
        self.query_vectorizer = query_vectorizer
        
    def _vectorize_query(self, queries: Iterable[str]):
        tokenized = map(self.tokenizer.tokenize, queries)
        indices = torch.LongTensor(map(self.query_vocab.token_seq_to_indices, tokenized))
        return self.query_vectorizer.forward(indices), tokenized

    def forward(self, queries: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        vectorized, tokenized = self._vectorize_query(queries)
        return self.internal_encoder(vectorized)

class VectorSeqEncoder(nn.Module, ABC):
    def __init__(self, input_dims):
        """
        Args:
            input_dims: Diminsionality of input vectors
        """
        super().__init__()
        self.input_dims = input_dims


    @abstractmethod
    def forward(self, seqs: torch.Tensor, input_lengths=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seqs: A tensor of dims (batch_size, seq_len, input_dims)
            input_lengths: Optinoal input lengths of the mini-batch. Used for
                padding if enabled.

        Returns:
            A tuple of tensors. The first is the of dims (batch_size, dim_size)
            which represents a "summarization" of the entire query's meaning.
            The second is a tensor of size (batch_size, seq_len, hidden_size)
            which represents a contextualized encoding for every token in the
            input (which could be used for something like an attention or copy
            mechanism in the decoder).
        """
        raise NotImplemented()

class RNNSeqEncoder(VectorSeqEncoder):
    """
    A module which takes in a sequence of vectors of size
    (batch, seq_len, input_dims) and encodes it using a RNN.

    This code was adapted partially from
    https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/EncoderRNN.py
    which is (C) IBM Corporation and available under http://www.apache.org/licenses/LICENSE-2.0

    Args:
        input_dims: Diminsionality of input vectors
        hidden_size: How many dims to use inside rnn cells
        rnn_cell: A type object of the rnn module to use (likely either nn.GRU or nn.LSTM)
        num_layers: Number of RNN layers to use
        input_dropout_p: A dropout over the inputs
        dropout_p: Dropout inside rnn_cell layers
        variable_lengths: Whether to add in packing
    """
    def __init__(
        self,
        input_dims: int,
        hidden_size: int,
        rnn_cell: Type = nn.GRU,
        num_layers: int = 1,
        input_dropout_p: float = 0,
        dropout_p: float = 0,
        bidirectional: bool = True,
        variable_lengths = False
    ):
        super().__init__(input_dims)
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = self.rnn_cell(input_dims, hidden_size, num_layers, num_layers,
                                 batch_first=True, dropout=dropout_p,
                                 bidirectional=bidirectional)

    def forward(self, seqs: torch.Tensor, input_lengths=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE (DNGros): this was just copied from the IBM implementation.
        # I am not sure exactly how this works. Look more into before using
        # variable lengths.
        seqs = self.input_dropout(seqs)
        if self.variable_lengths:
            seqs = nn.utils.rnn.pack_padded_sequence(seqs, input_lengths, batch_first=True)
        output, hidden = self.rnn(seqs)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            raise NotImplemented("Make sure this is working as expected")
        # As the "summary value" we return the last value in the sequence
        return output[:, -1, :], output


def make_default_query_encoder(
    x_tokenizer: Tokenizer,
    query_vocab: Vocab,
    query_vectorizer: VectorizerBase,
    output_size = 64
) -> QueryEncoder:
    """Factory for making a default QueryEncoder"""
    internal_encoder = RNNSeqEncoder(
        query_vectorizer.feature_len(),
        output_size,
    )
    return StringQueryEncoder(x_tokenizer, query_vocab, query_vectorizer, internal_encoder)


