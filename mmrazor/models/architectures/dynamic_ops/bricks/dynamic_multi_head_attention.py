# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Dict, Set, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from torch import Tensor

from mmrazor.models.architectures.ops import MultiheadAttention
from mmrazor.models.mutables.base_mutable import BaseMutable
from ..mixins import DynamicChannelMixin
from .dynamic_relative_position import DynamicRelativePosition2D  # noqa: E501


class DynamicMultiheadAttention(MultiheadAttention, DynamicChannelMixin):
    """Dynamic Multihead Attention with iRPE..

    Note:
        Arguments for ``__init__`` of ``DynamicMultiheadAttention`` is
        totally same as
        :obj:`mmrazor.models.architectures.MultiheadAttention`.
    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `num_heads`、 `embed_dims`、 `q_embed_dims`.
            The key of the dict must in ``accepted_mutable_attrs``.
    """

    mutable_attrs: nn.ModuleDict
    relative_position: bool
    max_relative_position: int
    w_qs: nn.Linear
    w_ks: nn.Linear
    w_vs: nn.Linear
    embed_dims: int
    q_embed_dims: int
    proj: nn.Linear
    attn_drop_rate: float
    accepted_mutable_attrs: Set[str] = {
        'num_heads', 'embed_dims', 'q_embed_dims'
    }
    attr_mappings: Dict[str, str] = {
        'in_channels': 'embed_dims',
        'out_channels': 'q_embed_dims',
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

        # dynamic image relative position encoding
        if self.relative_position:
            self.rel_pos_embed_k = DynamicRelativePosition2D(
                self.head_dims, self.max_relative_position)
            self.rel_pos_embed_v = DynamicRelativePosition2D(
                self.head_dims, self.max_relative_position)

    @property
    def mutable_num_heads(self):
        """Mutable number of heads."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['num_heads']

    @property
    def mutable_embed_dims(self):
        """Mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['embed_dims']

    @property
    def mutable_q_embed_dims(self):
        """Mutable intermediate embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['q_embed_dims']

    def register_mutable_attr(self, attr: str, mutable: BaseMutable):
        """Register attribute of mutable."""
        self.check_mutable_attr_valid(attr)
        if attr in self.attr_mappings:
            attr_map = self.attr_mappings[attr]
            assert attr_map in self.accepted_mutable_attrs
            # if hasattr(self, 'mutable_attrs'):
            if attr_map in self.mutable_attrs:
                print_log(
                    f'{attr_map}({attr}) is already in `mutable_attrs`',
                    level=logging.WARNING)
            else:
                self._register_mutable_attr(attr_map, mutable)
        elif attr in self.accepted_mutable_attrs:
            self._register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError

    def _register_mutable_attr(self, attr: str, mutable: BaseMutable):
        """Register `embed_dims` `q_embed_dims` `num_heads`"""
        if attr == 'num_heads':
            self._register_mutable_num_heads(mutable)
        elif attr == 'embed_dims':
            self._register_mutable_embed_dims(mutable)
        elif attr == 'q_embed_dims':
            self._register_mutable_q_embed_dims(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_num_heads(self, mutable_num_heads):
        """Register the mutable number of heads."""
        assert hasattr(self, 'mutable_attrs')
        current_choice = mutable_num_heads.current_choice
        if current_choice > self.num_heads:
            raise ValueError(
                f'Expect value of mutable to be smaller or equal than '
                f'{self.num_heads} as `num_heads`, but got: {current_choice}.')

        self.mutable_attrs['num_heads'] = mutable_num_heads

    def _register_mutable_embed_dims(self, mutable_embed_dims):
        """Register mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        mask_size = mutable_embed_dims.current_mask.size(0)
        if mask_size != self.embed_dims:
            raise ValueError(
                f'Expect mask size of mutable to be {self.embed_dims} as '
                f'`embed_dims`, but got: {mask_size}.')

        self.mutable_attrs['embed_dims'] = mutable_embed_dims

    def _register_mutable_q_embed_dims(self, mutable_q_embed_dims):
        """Register intermediate mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        self.mutable_attrs['q_embed_dims'] = mutable_q_embed_dims

    def _get_dynamic_proj_params(self, w: nn.Linear) -> Tuple[Tensor, Tensor]:
        """Get parameters of dynamic projection.

        Note:
            The input dimension is decided by `mutable_q_embed_dims`.
            The output dimension is decided by `mutable_embed_dims`.
        """
        # TODO support mask
        if self.mutable_embed_dims is None and \
                self.mutable_q_embed_dims is None:
            return w.weight, w.bias

        if self.mutable_q_embed_dims is not None:
            in_features = self.mutable_q_embed_dims.activated_channels
        else:
            in_features = self.embed_dims

        if self.mutable_embed_dims is not None:
            out_features = self.mutable_embed_dims.activated_channels
        else:
            out_features = self.embed_dims

        weight = w.weight[:out_features, :in_features]
        bias = w.bias[:out_features] if w.bias is not None else None

        return weight, bias

    def _get_dynamic_qkv_params(self, w: nn.Linear) -> Tuple[Tensor, Tensor]:
        """Get parameters of dynamic QKV.

        Note:
            The output dimension is decided by `mutable_q_embed_dims`.
            The input dimension is decided by `mutable_embed_dims`.
        """
        # TODO support mask later
        if self.mutable_q_embed_dims is None and \
                self.mutable_embed_dims is None:
            return w.weight, w.bias

        if self.mutable_embed_dims is not None:
            in_features = self.mutable_embed_dims.activated_channels
        else:
            in_features = self.embed_dims

        if self.mutable_q_embed_dims is not None:
            out_features = self.mutable_q_embed_dims.activated_channels
        else:
            out_features = self.mutable_q_embed_dims

        weight = w.weight[:out_features, :in_features]
        bias = w.bias[:out_features] if w.bias is not None else None

        return weight, bias

    def to_static_op(self) -> MultiheadAttention:
        """Convert dynamic MultiheadAttention to static one."""
        self.check_if_mutables_fixed()

        embed_dims = self.mutable_embed_dims.activated_channels
        num_heads = self.mutable_num_heads.current_choice

        q_w, q_b = self._get_dynamic_qkv_params(self.w_qs)
        k_w, k_b = self._get_dynamic_qkv_params(self.w_ks)
        v_w, v_b = self._get_dynamic_qkv_params(self.w_vs)

        proj_w, proj_b = self._get_dynamic_proj_params(self.proj)

        static_mha = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            input_dims=None,
            relative_position=self.relative_position,
            max_relative_position=self.max_relative_position)

        static_mha.w_qs.weight = nn.Parameter(q_w.clone())
        static_mha.w_qs.bias = nn.Parameter(q_b.clone())

        static_mha.w_ks.weight = nn.Parameter(k_w.clone())
        static_mha.w_ks.bias = nn.Parameter(k_b.clone())

        static_mha.w_vs.weight = nn.Parameter(v_w.clone())
        static_mha.w_vs.bias = nn.Parameter(v_b.clone())

        static_mha.proj.weight = nn.Parameter(proj_w.clone())
        static_mha.proj.bias = nn.Parameter(proj_b.clone())

        if self.relative_position:
            static_mha.rel_pos_embed_k = self.rel_pos_embed_k.to_static_op()
            static_mha.rel_pos_embed_v = self.rel_pos_embed_v.to_static_op()

        return static_mha

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
