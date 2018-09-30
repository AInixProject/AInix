import torch
from torch import Tensor
import numpy as np
import math


class EmbeddedAppender(torch.nn.Module):
    """A model that simply concats on a single learned value to every value
    in the sequence.

    Args:
        dim_to_append: The length of the vector we will learn to append on.
    """
    def __init__(self, dims_to_append: int):
        super().__init__()
        self.dims_to_append = dims_to_append
        self.learned_vec = torch.nn.Parameter(torch.randn(dims_to_append))

    def forward(self, base_values: Tensor) -> torch.Tensor:
        expanded_learned_vec = self.learned_vec.expand(*base_values.shape[:2], -1)
        return torch.cat((base_values, expanded_learned_vec), 2)


class EmbeddedAdder(torch.nn.Module):
    """A model that simply adds on a single learned value to every value
    in the sequence.

    Args:
        len_vec: The length of the vector we will learn to add to.
    """
    def __init__(self, vec_len: int):
        super().__init__()
        self.vec_len = vec_len
        self.learned_vec = torch.nn.Parameter(torch.randn(vec_len))

    def forward(self, base_values: Tensor) -> torch.Tensor:
        expanded_learned_vec = self.learned_vec.expand(*base_values.shape[:2], -1)
        return base_values + expanded_learned_vec


def _gen_timing_signal(batches, length, channels, min_timescale=1.0,
                       max_timescale=1.0e4, positions=None):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/kolloldas/torchnlp/blob/master/torchnlp/modules/transformer/main.py
    """
    if not positions:
        positions = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])
    t_signal = torch.from_numpy(signal).type(torch.FloatTensor)
    t_signal.requires_gradient = False

    return t_signal.expand(batches, -1, -1)


class TimingSignalConcat(torch.nn.Module):
    """Concats on in a timeing signal based off the a series of sinusoids. This
    can be used an input into a Transformer module.
    Transformer paper ref: https://arxiv.org/abs/1706.03762

    Args:
        signal_dims: length of the vector to make timing signal for and concat on.
        min_timescale: min period scaling of signal
        max_timescale: max period scaling of signal
    """
    def __init__(
            self,
            signal_dims: int,
            min_timescale=1.0,
            max_timescale=1000.0
    ):
        super().__init__()
        self.signal_dims = signal_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, base_values: Tensor, positions=None) -> torch.Tensor:
        signal = _gen_timing_signal(*base_values.shape[:2], self.signal_dims, positions=positions)
        return torch.cat((base_values, signal), 2)


class TimingSignalAdd(torch.nn.Module):
    """Adds timeing made up of a series of sinusoids into a tensor of dens vectors. This
    can be used an input into a Transformer module.
    Transformer paper ref: https://arxiv.org/abs/1706.03762

    Args:
        min_timescale: min period scaling of signal
        max_timescale: max period scaling of signal
    """
    def __init__(
        self,
        min_timescale=1.0,
        max_timescale=1000.0
    ):
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, base_values: Tensor, positions=None) -> torch.Tensor:
        # TODO (DNGros): consider just generate a long one and slicing from it.
        signal = _gen_timing_signal(*base_values.shape[:3], positions=positions)
        #print("signal", signal)
        return base_values + signal

