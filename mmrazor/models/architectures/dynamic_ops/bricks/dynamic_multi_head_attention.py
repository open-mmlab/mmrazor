# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicMHAMixin
from mmrazor.models.architectures.ops import MultiheadAttention
from mmrazor.models.mutables.base_mutable import BaseMutable
from .dynamic_relative_position import DynamicRelativePosition2D  # noqa: E501


class DynamicMultiheadAttention(MultiheadAttention, DynamicMHAMixin):
    """Dynamic Multihead Attention with iRPE."""

    accepted_mutable_attrs = {
        'num_heads',
        'embed_dims',
        'q_embed_dims',
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str,
                                 BaseMutable] = nn.ModuleDict()  # type: ignore

        # dynamic image relative position encoding
        if self.relative_position:
            self.rel_pos_embed_k = DynamicRelativePosition2D(
                self.head_dims, self.max_relative_position)
            self.rel_pos_embed_v = DynamicRelativePosition2D(
                self.head_dims, self.max_relative_position)

    @property
    def in_channels(self):
        return getattr(self, self.attr_mappings['in_channels'])

    @property
    def out_channels(self):
        return getattr(self, self.attr_mappings['out_channels'])

    @classmethod
    def convert_from(cls, module):
        """Convert the static module to dynamic one."""
        dynamic_mha = cls(
            embed_dims=module.embed_dims,
            num_heads=module.num_heads,
        )
        return dynamic_mha

    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return MultiheadAttention

    def forward(self, x: Tensor) -> Tensor:
        """Forward of dynamic MultiheadAttention."""
        B, N = x.shape[0], x.shape[1]
        q_w, q_b = self._get_dynamic_qkv_params(self.w_qs)
        k_w, k_b = self._get_dynamic_qkv_params(self.w_ks)
        v_w, v_b = self._get_dynamic_qkv_params(self.w_vs)

        q_embed_dims = self.mutable_q_embed_dims.activated_channels
        num_heads = self.mutable_num_heads.current_choice

        q = F.linear(x, q_w, q_b).view(B, N, num_heads,
                                       q_embed_dims // num_heads)
        k = F.linear(x, k_w, k_b).view(B, N, num_heads,
                                       q_embed_dims // num_heads)
        v = F.linear(x, v_w, v_b).view(B, N, num_heads,
                                       q_embed_dims // num_heads)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, num_heads * B, -1)  # noqa: E501
                           @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, num_heads, N, N) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * num_heads, -1)
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(
                B, num_heads, N, -1).transpose(2, 1).reshape(B, N, -1)

        # proj
        weight, bias = self._get_dynamic_proj_params(self.proj)
        x = F.linear(x, weight, bias)
        x = self.proj_drop(x)
        return x
