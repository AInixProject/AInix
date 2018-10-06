"""Vectorizers convert a elements of a vocab into a tensor of dense vectors.

Vectorizers can be composed together to do more complex Vectorizing."""
from abc import ABC, abstractmethod
from collections import Counter

import attr
from ainix_common.parsing.parseast import ObjectChoiceNode
from ainix_kernel.indexing.examplestore import Example
from ainix_kernel.model_util.tokenizers import Tokenizer
from ainix_kernel.model_util.vocab import Vocab
import torch
from ainix_kernel.models.model_types import Pretrainable, Pretrainer
from typing import Callable, Iterable, Tuple


class VectorizerBase(torch.nn.Module, Pretrainable):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def feature_len(self) -> int:
        """Returns the length of the feature vector this outputs"""
        pass

    @abstractmethod
    def forward(self, indicies) -> torch.Tensor:
        """Converts a tensor of indicies to vectors.

        Args:
            indicies: Long Tensor of the form (batch, sequence_len).

        Returns:
            Float tensor of form (batch, sequence_len, feature_len)
        """
        pass


class TorchDeepEmbed(VectorizerBase):
    """A Vectorizer that keeps track of pytorch vectors for embedding tokens

    Args:
        vocab: the vocab that we wish to embed (used to determine the number of
            tokens in the embedding)
        embed_dim: The number of dimensions of the vectorspace
    """
    def __init__(self, vocab: Vocab, embed_dim: int):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.embed = torch.nn.Embedding(len(vocab), embed_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Converts a tensor of indicies to vectors which are embeddings into
        a backpropigation-learned vector space.

        Args:
            indices: Long Tensor of the indicies into the vocab (batch, sequence_len).

        Returns:
            Float tensor of form (batch, sequence_len, feature_len)
        """
        return self.embed(indices)

    def feature_len(self):
        return self.embed_dim

####
# Pretraining Vectorizers
####

@attr.s(auto_attribs=True)
class AverageProcFuncReturnVal:
    indicies_to_update: torch.Tensor
    additional_counts: torch.Tensor
    additional_sums: torch.Tensor


AVERAGING_PROCESSING_FUNC_TYPE = \
    Callable[[Example, ObjectChoiceNode], AverageProcFuncReturnVal]


class TorchAveragingPretrainer(Pretrainer):
    """A pretrainer which alters sum and counts based off some callable."""
    def __init__(
        self,
        counts: torch.Tensor,
        average_store: torch.Tensor,
        process_func: AVERAGING_PROCESSING_FUNC_TYPE,
    ):
        super().__init__()
        self._average_store = average_store
        self._counts = counts
        self._sums = self._counts.reshape(-1, 1) * self._average_store
        self.process_func = process_func

    def pretrain_example(self, example: Example, y_ast: ObjectChoiceNode):
        # TODO (DNGros): There are probably more numerically stable ways of
        # doing a running average. Maybe look into those eventually...
        super().pretrain_example(example, y_ast)
        proc = self.process_func(example, y_ast)
        self._sums[proc.indicies_to_update] += proc.additional_sums
        self._counts[proc.indicies_to_update] += proc.additional_counts

    def close(self):
        super().close()
        self._average_store.data = self._sums / self._counts.reshape(-1, 1)



class PretrainedAvgVectorizer(VectorizerBase):
    """A vectorizer that records and produces averages of some function applied
    during pretraining."""
    def __init__(
        self,
        dims: int,
        process_func: AVERAGING_PROCESSING_FUNC_TYPE,
        vocab: Vocab
    ):
        super().__init__()
        self.process_func = process_func
        self._dims = dims
        self.counts = None
        self.averages = None
        self.vocab = vocab

    def feature_len(self):
        return self._dims

    def resize_if_needed(self):
        """Resizes internal representations if vocab size has changed"""
        # TODO (DNGros): This may not be a good idea as it means that we have
        # to fully update the vocab before extending. Idk... We'll figure
        # it out when we get to supporting extending vocabs and stuff
        if self.averages is None:
            self.counts = torch.zeros(len(self.vocab))
            self.averages = torch.zeros(len(self.vocab), self._dims)
            assert not self.averages.requires_grad
        else:
            raise NotImplemented("Haven't actually implemented resizing")

    def get_pretrainer(self):
        self.resize_if_needed()
        return TorchAveragingPretrainer(self.counts, self.averages, self.process_func)

    def forward(self, indicies) -> torch.Tensor:
        return self.averages[indicies]


def make_xquery_avg_pretrain_func(
    ast_vocab: Vocab,
    ast_tokenizer: Tokenizer,
    x_vec_func: Callable[[str], torch.Tensor],
) -> AVERAGING_PROCESSING_FUNC_TYPE:
    """A a closure that produces a pretrain averaging function that applies some
    function onto the xquery string of the examples"""
    def out_func(example: Example, y_ast: ObjectChoiceNode) -> AverageProcFuncReturnVal:
        x_vectorized = x_vec_func(example.xquery)
        y_ast_indicies = ast_vocab.token_seq_to_indices(
            ast_tokenizer.tokenize(y_ast), as_torch=False)
        items = list(Counter(y_ast_indicies).items())
        update_indicies = torch.LongTensor([x[0] for x in items])
        update_counts = torch.Tensor([x[1] for x in items])
        update_sums = update_counts.reshape(-1, 1) * x_vectorized
        return AverageProcFuncReturnVal(update_indicies, update_counts, update_sums)

    return out_func

