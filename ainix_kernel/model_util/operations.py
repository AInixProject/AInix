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


class MultilabelKindaCategoricalCrossEntropy(torch.nn.Module):
    """A somewhat funky loss function that sort of seems like a good idea.
    It is designed to accommodate when you have multiple correct labels
    but also a weighted preference between each label. It cares more about
    the max being a one of the valid labels, over the weight ordering between
    the labels.

    The basic idea is for each of the valid labels l_i, calculate the
    CatagoricalCrossEntropyLoss of l_i with all other valid labels zeroed out
    then multiply that loss by the weight of l_i.

    This means the valid labels are not
    """
    pass  # not implemented

