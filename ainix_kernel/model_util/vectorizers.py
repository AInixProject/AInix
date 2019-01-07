"""Vectorizers convert a elements of a vocab into a tensor of dense vectors.

Vectorizers can be composed together to do more complex Vectorizing."""
from abc import abstractmethod
from collections import Counter

import attr
from ainix_common.parsing.ast_components import ObjectChoiceNode
from ainix_kernel.indexing.examplestore import Example
from ainix_common.parsing.model_specific.tokenizers import Tokenizer
from ainix_kernel.model_util.vocab import Vocab
import torch
from ainix_kernel.models.model_types import Pretrainable, Pretrainer
from typing import Callable
import numpy as np

SAVE_STATE_NAME_KEY = "vectorizer_name"


class VectorizerBase(torch.nn.Module, Pretrainable):
    SAVE_STATE_NAME_VALUE = None

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

    def get_save_state_dict(self):
        raise NotImplemented()


class TorchDeepEmbed(VectorizerBase):
    """A Vectorizer that keeps track of pytorch vectors for embedding tokens

    Args:
        vocab: the vocab that we wish to embed (used to determine the number of
            tokens in the embedding)
        embed_dim: The number of dimensions of the vectorspace
    """
    SAVE_STATE_NAME_VALUE = "TorchDeepEmbed"

    def __init__(self, vocab_size: object, embed_dim: object) -> object:
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.vocab_size = vocab_size

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

    def get_save_state_dict(self):
        return {
            SAVE_STATE_NAME_KEY: self.SAVE_STATE_NAME_VALUE,
            "model_state": self.state_dict(),
            "embed_dim": self.embed_dim,
            "vocab_size": self.vocab_size,
            "version": 0
        }

    @classmethod
    def create_from_save_state_dict(
        cls,
        save_dict: dict,
        dirty_indices_mask: np.ndarray
    ) -> 'TorchDeepEmbed':
        """
        Args:
            dirty_indices_mask: A binary array of length of the new vocab size.
                Should be True if the vocab index is "dirty" and False otherwise.
                A dirty index happens for tokens which are different than they
                were origionally when saved, or for new tokens which didn't used
                to exist

        Returns:

        """
        if dirty_indices_mask is not None:
            # TODO (DNGros): actually do this
            raise NotImplemented()
        instance = TorchDeepEmbed(save_dict['vocab_size'], save_dict['embed_dim'])
        instance.load_state_dict(save_dict['model_state'])
        return instance


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


def vectorizer_from_save_dict(save_dict: dict) -> VectorizerBase:
    """Creates a vectorizer based of perviously seriallized save_state_dict.
    These are creating using a method on each vectorizer. This method handles
    detecting the kind of vectorizer that the save_dict came from and
    instantiating the right one."""
    name = save_dict[SAVE_STATE_NAME_KEY]
    # TODO: figure out vocab dirty masks
    if name == TorchDeepEmbed.SAVE_STATE_NAME_VALUE:
        return TorchDeepEmbed.create_from_save_state_dict(save_dict, None)
    else:
        raise ValueError(f"Name {name} not recocognized")
