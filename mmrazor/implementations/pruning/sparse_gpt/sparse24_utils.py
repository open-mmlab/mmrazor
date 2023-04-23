# Copyright (c) OpenMMLab. All rights reserved.
import torch


@torch.no_grad()
def is_weight_sparse_24(weight: torch.Tensor, dim=-1):
    """"Check if the weight is sparse 24."""
    weight = weight.transpose(-1, dim).reshape(-1, 4)  # N 4
    is_zero = (weight == 0).sum(-1)  # N
    return (is_zero >= 2).all()
