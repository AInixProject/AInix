from typing import Tuple, List, Callable

import torch
from torch import nn
from torch.nn import LeakyReLU

from ainix_kernel.models.multiforward import add_hooks, MultiforwardTorchModule


class Multiembedder(MultiforwardTorchModule):
    """A embedder which is a combination of multiple embedd index. This was created
    to be able to handle the multiple embeddings required in modded string tokens (
    for example one embedding for string, one for case, and one for whitespace all
    combined). It could be used for other things as well.

    Rather than just concatenating together the embeddings or adding them together,
    this does a combination of the two. The reasoning is that there probably isn't
    an interdependence between all features, thus a concatenation is inefficient.
    On the otherhand, there is also at least some nonlinear interpendence, so a
    simple adding won't do.

    For a user defined portion of the features we add, and for the other portion
    we concat together and pass through a feed forward layer.
    """
    def __init__(
        self,
        vocab_sizes: Tuple[int, ...],
        target_out_len: int,
        additive_portion: float = 0.75,
        combiner_activation_factory: Callable = lambda: LeakyReLU()
    ):
        super().__init__()
        self.num_categories = len(vocab_sizes)
        self.target_out_len = target_out_len
        assert 0 <= additive_portion <= 1
        self.additive_portion = additive_portion
        self.embedders = nn.ModuleList(
            [nn.Embedding(v_size, target_out_len) for v_size in vocab_sizes])
        eps = 0.00001
        assert -eps < int(additive_portion*target_out_len) - additive_portion*target_out_len < eps
        combine_part_size = int(target_out_len*(1-additive_portion))
        self.add_size = int(additive_portion*target_out_len)
        self.combine_linear = nn.Linear(combine_part_size*self.num_categories, combine_part_size)
        self.combine_activation = combiner_activation_factory() \
            if combiner_activation_factory else None

    @add_hooks
    def embed(self, features: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            features: LongTensor of dim (category_number, batch_size, seq_length)
                category_number refers to the 0th, 1st, ... embedding.

        Returns:
            A tensor of dim (batch-size, seq_length, target_out_length)
        """
        if len(features) != self.num_categories:
            raise ValueError("Unexpected num of categories")
        add_components: List[torch.Tensor] = []
        combined_compoents: List[torch.Tensor] = []
        for embedder, category in zip(self.embedders, features):
            embed = embedder(category)
            add_components.append(embed[:, :, :self.add_size])
            combined_compoents.append(embed[:, :, self.add_size:])
        added_part = torch.sum(torch.stack(tuple(add_components)), dim=0)
        combined_parts_concatted = torch.cat(tuple(combined_compoents), dim=2)
        combined_reduced = self.combine_linear(combined_parts_concatted)
        if self.combine_activation:
            combined_reduced = self.combine_activation(combined_reduced)
        return torch.cat((added_part, combined_reduced), dim=2)
