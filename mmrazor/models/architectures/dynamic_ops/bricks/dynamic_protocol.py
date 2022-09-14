# Copyright (c) OpenMMLab. All rights reserved.

import sys

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import torch.nn as nn

from mmrazor.models.architectures.ops import RelativePosition2D
from mmrazor.models.mutables.base_mutable import BaseMutable


class DynamicMHAProtocol(Protocol):
    """Protocol for Dynamic Multi head Attention."""
    relative_position: bool
    max_relative_position: int
    rel_pos_embed_k: RelativePosition2D
    rel_pos_embed_v: RelativePosition2D
    w_qs: nn.Linear
    w_ks: nn.Linear
    w_vs: nn.Linear
    embed_dims: int
    proj: nn.Module
    attn_drop_rate: float


class DynamicRPProtocol(Protocol):
    """Protocol for Dynamic Relative Position."""
    mutable_attrs: nn.ModuleDict
    mutable_head_dims: BaseMutable
    head_dims: int
    max_relative_position: int
    embeddings_table_v: nn.Parameter
    embeddings_table_h: nn.Parameter
