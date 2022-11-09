# Copyright (c) OpenMMLab. All rights reserved.
import logging
import sys
from typing import Dict, Set, Tuple

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import torch.nn as nn
from mmengine import print_log
from torch import Tensor

from mmrazor.models.architectures.ops import (MultiheadAttention,
                                              RelativePosition2D)
from mmrazor.models.mutables.base_mutable import BaseMutable
# from .dynamic_mixins import DynamicMixin
from .dynamic_mixins import DynamicChannelMixin


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
    q_embed_dims: int
    proj: nn.Linear
    attn_drop_rate: float
    mutable_attrs: Dict


class DynamicMHAMixin(DynamicChannelMixin, DynamicMHAProtocol):
    """Mixins for Dynamic Multi head attention.

    Note:
        `embed_dims` serve the in_dim of qkv and out_dim of proj
        `q_embed_dims` serve the out_dim of qkv and in_dim of proj
        `q_embed_dims` is a DerivedMutable derived from `num_heads`
            with `num_heads` \times 64.
    """
    accepted_mutable_attrs: Set[str] = {
        'num_heads', 'embed_dims', 'q_embed_dims'
    }

    attr_mappings: Dict[str, str] = {
        'in_channels': 'embed_dims',
        'out_channels': 'q_embed_dims',
    }

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

    def to_static_op(self) -> nn.Module:
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
            attn_drop_rate=self.attn_drop_rate,
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
