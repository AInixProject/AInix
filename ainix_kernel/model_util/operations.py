"""Useful operations on torch tensors which are not in the Pytorch lib"""
from typing import Tuple

import torch


def manual_bincount(groups: torch.Tensor, weights: torch.Tensor = None):
    """A manual version of torch.bincount since as of torch 1.0.0 backprop isn't
    supported on bincount"""
    max_group = int(torch.max(groups))
    counts = torch.zeros((max_group + 1,))
    for g in range(max_group + 1):
        matching = groups == g
        if weights is None:
            counts[g] = torch.sum(matching)
        else:
            counts[g] = torch.sum(weights[matching])
    return counts


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
    # Actual torch bincount apparently doesn't support backprop
    #reduced_vals = torch.bincount(group_ind_for_each_val, values)
    reduced_vals = manual_bincount(group_ind_for_each_val, values)
    return reduced_vals, group_keys


def pack_picks(data, picks):
    """Flattens a iterable of tensors. For each selection, we take indicies from
    a corresponding part of another and packs it all together"""
    return torch.cat([data_b[pick_b] if len(pick_b) > 0 else data_b.new()
                      for data_b, pick_b in zip(data, picks)])


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


#class BackwardsMask(torch.autograd.Function):
#    """An identity function, but lets only certain functions be non-zero"""
#    @staticmethod
#    def forward(ctx, input, select_mask):
#        pass
#
#    @staticmethod
#    def forward(ctx, input, select_mask):
#        pass

