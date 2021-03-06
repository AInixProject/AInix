from abc import ABC, abstractmethod

import torch
from torch import nn
from typing import Tuple, Type, Sequence, List, Union, Optional

from ainix_common.parsing.model_specific.tokenizers import ModifiedStringToken, WhitespaceModifier, \
    SpaceTokenizer, add_str_pads, get_default_pieced_tokenizer_word_list, add_pads_to_mod_tokens
from ainix_kernel.model_util import vectorizers
from ainix_kernel.model_util.glove import get_glove_words
from ainix_kernel.model_util.stop_words import STOP_WORDS
from ainix_kernel.model_util.vectorizers import VectorizerBase, vectorizer_from_save_dict
from ainix_kernel.model_util.vocab import Vocab, BasicVocab, make_x_vocab_from_examples
from ainix_common.parsing.model_specific import tokenizers, parse_constants
import numpy as np


class QueryEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        queries: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[ModifiedStringToken]]]:
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

    @abstractmethod
    def get_tokenizer(self) -> tokenizers.Tokenizer:
        pass

    def get_save_state_dict(self):
        raise NotImplemented

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict
    ) -> 'QueryEncoder':
        raise NotImplemented


class StringQueryEncoder(QueryEncoder):
    """NOTE this code somewhat legacy and needs to be refactored to work better
    with the QueryEncoder in cookiemonster.py"""
    def __init__(
        self,
        tokenizer: tokenizers.StringTokenizer,
        query_vocab: Vocab,
        query_vectorizer: VectorizerBase,
        internal_encoder: 'VectorSeqEncoder'
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.query_vocab = query_vocab
        self.query_vectorizer = query_vectorizer
        self.internal_encoder = internal_encoder
        
    def _vectorize_query(self, queries: Sequence[str]):
        """Converts a batch of string queries into dense vectors"""
        tokenized = self.tokenizer.tokenize_batch(queries, take_only_tokens=True)
        tokenized, input_lens = tokenizers.add_str_pads(tokenized)
        tokenized = np.array(tokenized)
        indices = self.query_vocab.token_seq_to_indices(np.array(tokenized))
        return self.query_vectorizer.forward(indices), tokenized, torch.LongTensor(input_lens)

    def forward(
        self,
        queries: Sequence[Sequence[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        vectorized, tokenized, input_lens = self._vectorize_query(queries)
        summary, memory = self.internal_encoder(vectorized, input_lens)
        return summary, memory, tokenized

    def get_tokenizer(self) -> tokenizers.StringTokenizer:
        return self.tokenizer

    def get_save_state_dict(self):
        return {
            "version": 0,
            "tokenizer": self.tokenizer.get_save_state_dict(),
            "query_vocab": self.query_vocab.get_save_state_dict(),
            "query_vectorizer": self.query_vectorizer.get_save_state_dict(),
            #"internal_encoder": self.internal_encoder.get_save_state_dict()
            # feeling lazy right now so just going to pickle the internal encoder
            # TODO (DNGros): Make custom handling of serialize deserialize
            "internal_encoder": self.internal_encoder
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict
    ) -> 'StringQueryEncoder':
        return StringQueryEncoder(
            tokenizer=tokenizers.tokenizer_from_save_dict(state_dict['tokenizer']),
            query_vocab=BasicVocab.create_from_save_state_dict(state_dict['query_vocab']),
            query_vectorizer=vectorizer_from_save_dict(state_dict['query_vectorizer']),
            internal_encoder=state_dict['internal_encoder']
        )


class ModTokensEncoder(nn.Module, ABC):
    def forward(
        self,
        token_inds: torch.LongTensor,
        case_mod_inds: torch.LongTensor,
        whitespace_mod_inds: torch.LongTensor,
        input_lens: Optional[torch.LongTensor] = None
    ):
        raise NotImplemented()

    @property
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplemented()

    def get_tokens_input_weights(self) -> torch.Tensor:
        """For the BERT task the prediction of the tokens shares the weights with
        base embedding. Get this weight"""
        raise NotImplemented


class VectorSeqEncoder(nn.Module, ABC):
    def __init__(self, input_dims):
        """
        Args:
            input_dims: Dimensionality of input vectors
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

    def get_save_state_dict(self):
        raise NotImplemented()

    @classmethod
    def create_from_save_state_dict(
        cls,
        state_dict: dict
    ) -> 'VectorSeqEncoder':
        raise NotImplemented()


def reorder_based_off_len(input_lens: torch.LongTensor, vals_to_reorder: Tuple[torch.Tensor]):
    """Reorder a batch of values in descending order of lengths. Used to make
    packing happy."""
    sorted_lens, sorting_inds = torch.sort(input_lens, descending=True)
    sorted_lens = sorted_lens.long()
    vals_after_reorder = [v[sorting_inds] for v in vals_to_reorder]
    return sorted_lens, sorting_inds, vals_after_reorder


def undo_len_ordering(sorting_inds, vals_to_undo: Tuple[torch.Tensor]):
    """Applies the inverse of reordering based off lengths. Requires the origional
    indicies that were used to index the ordering of the origional values"""
    vals_after_undo = []
    for v in vals_to_undo:
        unsorted = v.new(*v.size())
        expanded_inds = sorting_inds
        for unsqueeze_count in range(len(v.shape) - 1):
            expanded_inds = expanded_inds.unsqueeze(1)
        expanded_inds = expanded_inds.expand(*v.shape)
        unsorted.scatter_(0, expanded_inds, v)
        vals_after_undo.append(unsorted)
    return vals_after_undo


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
        summary_size: How many dims for the summary of the whole sequence. This
            is derived from the last val in the sequence. If None, then a summary is
            not computed.
        rnn_cell: A type object of the rnn module to use (likely either nn.GRU or nn.LSTM)
        num_layers: Number of RNN layers to use
        input_dropout_p: A dropout over the inputs
        dropout_p: Dropout inside rnn_cell layers
        variable_lengths: Whether to expect different lens
    """
    def __init__(
        self,
        input_dims: int,
        hidden_size: int,
        summary_size: Optional[int],
        memory_tokens_size: int,
        rnn_cell: Type = nn.GRU,
        num_layers: int = 1,
        input_dropout_p: float = 0.0,
        dropout_p: float = 0,
        bidirectional: bool = True,
        variable_lengths=False
    ):
        super().__init__(input_dims)
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = rnn_cell(input_dims, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_p,
                            bidirectional=bidirectional)
        # Actual hidden size will be twice size since bidirectional
        self.perform_summary = summary_size is not None
        if self.perform_summary:
            self.summary_linear = nn.Linear(hidden_size*2, summary_size)
        self.memory_tokens_linear = nn.Linear(hidden_size*2, memory_tokens_size)
        self.variable_lengths = variable_lengths

    def forward(
        self,
        seqs: torch.Tensor,
        input_lengths=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seqs = self.input_dropout(seqs)
        # Pepare if have variable lens
        assert (input_lengths is None) == (not self.variable_lengths)
        if self.variable_lengths:
            input_lengths, sort_inds, (seqs,) = reorder_based_off_len(input_lengths, (seqs,))
            seqs = nn.utils.rnn.pack_padded_sequence(seqs, input_lengths, batch_first=True)
        # Do actual processing
        outputs, final_hiddens = self.rnn(seqs)
        # Handle post processing if variable lengths
        if self.variable_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            input_lengths, outputs = undo_len_ordering(sort_inds, (input_lengths, outputs))
            last_val_of_each_in_batch = outputs[range(input_lengths.size(0)), input_lengths - 1]
        else:
            last_val_of_each_in_batch = outputs[:, -1]
        # Prepare output
        memory_tokens = self.memory_tokens_linear(outputs)
        if self.perform_summary:
            summaries = self.summary_linear(last_val_of_each_in_batch)
            return summaries, memory_tokens
        else:
            return memory_tokens


class SimpleGloveEncoder(QueryEncoder):
    def __init__(self, vocab: Vocab, dims: int=50):
        super().__init__()
        self._tokenizer, _ = get_default_pieced_tokenizer_word_list("glove_vocab.txt")
        self._glove_vecs = get_glove_words(dims, vocab)

    def forward(
        self,
        queries: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[ModifiedStringToken]]]:
        tokenized = self._tokenizer.tokenize_batch(queries, True)
        tokenized, original_lens = add_pads_to_mod_tokens(tokenized)
        summaries = []
        embedded_toks = []
        for batch in tokenized:
            this_toks = []
            this_sum = np.zeros(self._glove_vecs.dims)
            this_sum_count = 0
            for word in batch:
                word: str = word.token_string
                vec = self._glove_vecs.get_vec(word, always_return_vec=True)
                this_toks.append(vec)
                if word in self._glove_vecs and word not in STOP_WORDS:
                    # Only count words actually in vocab, not UNKs or PADs
                    this_sum += vec
                    this_sum_count += 1
            summaries.append(this_sum / this_sum_count)
            embedded_toks.append(this_toks)
        return torch.Tensor(summaries), torch.Tensor(embedded_toks), tokenized

    def get_tokenizer(self) -> tokenizers.Tokenizer:
        return self._tokenizer

