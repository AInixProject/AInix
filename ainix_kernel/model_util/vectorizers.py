"""Vectorizers convert a vector of vocab indicies into a tensor of dense vectors.

Vectorizers can be composed together to do more complex Vectorizing."""
from abc import ABC, abstractmethod
from ainix_kernel.model_util.vocab import Vocab
import torch


class VectorizerBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def feature_len(self) -> int:
        """Returns the length of the feature vector this outputs"""
        pass

    @abstractmethod
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
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

    def forward(self, indices) -> torch.Tensor:
        return self.embed(indices)

    def feature_len(self):
        return self.embed_dim


