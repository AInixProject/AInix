"""Useful operations on torch tensors which are not in the Pytorch lib"""
from typing import Tuple

import torch


def sparse_groupby_sum(
    values: torch.Tensor,
    value_keys: torch.Tensor,
    sort_out_keys: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a list of values and their corresponding keys, finds the sums for each key.
    The keys can be sparse.

    Args:
        values: the values we want to reduce
        value_keys: the group which each value belongs to. Should be same first dim as values.
        sort_out_keys: Whether to sort the group_keys on output
    Returns:
        reduced_vals: A tensor which is the sum for each group
        group_keys: A tensor which represents the key for each group. There will
            be one for each unique value in value_keys
    """
    # TODO (figure out dims / batching). Right now it flattens it
    group_keys, group_ind_for_each_val = torch.unique(value_keys, sort_out_keys, True)
    reduced_vals = torch.bincount(group_ind_for_each_val, values)
    return reduced_vals, group_keys

