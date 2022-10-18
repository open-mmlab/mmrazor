# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Dict, Set

import torch
from mmcls.models.utils import PatchEmbed
from mmengine import print_log
from torch import nn

from mmrazor.models.mutables.base_mutable import BaseMutable
from .dynamic_mixins import DynamicChannelMixin


class DynamicPatchEmbedMixin(DynamicChannelMixin):

    accepted_mutable_attrs: Set[str] = {'embed_dims'}
    attr_mappings: Dict[str, str] = {
        'in_channels': 'embed_dims',
        'out_channels': 'embed_dims'
    }

    @property
    def mutable_embed_dims(self):
        """Mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['embed_dims']

    def register_mutable_attr(self: PatchEmbed, attr: str,
                              mutable: BaseMutable):
        """Register attribute of mutable."""
        self.check_mutable_attr_valid(attr)
        if attr in self.attr_mappings:
            attr_map = self.attr_mappings[attr]
            assert attr_map in self.accepted_mutable_attrs
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

    def _register_mutable_attr(self, attr, mutable):
        """Register `embed_dims`."""
        if attr == 'embed_dims':
            self._register_embed_dims(mutable)
        else:
            raise NotImplementedError

    def _register_embed_dims(self: PatchEmbed,
                             mutable_patch_embedding: BaseMutable) -> None:
        """Register mutable embedding dimension."""
        mask_size = mutable_patch_embedding.current_mask.size(0)

        if mask_size != self.embed_dims:
            raise ValueError(
                f'Expect mask size of mutable to be {self.embed_dims} as '
                f'`embed_dims`, but got: {mask_size}.')

        self.mutable_attrs['embed_dims'] = mutable_patch_embedding

    def _get_dynamic_params(self: PatchEmbed) -> torch.Tensor:
        """Get mask of ``embed_dims``"""
        if 'embed_dims' not in self.mutable_attrs:
            return self.projection.weight, self.projection.bias
        else:
            out_mask = self.mutable_embed_dims.current_mask.to(
                self.projection.weight.device)
            weight = self.projection.weight[out_mask][:]
            bias = self.projection.bias[
                out_mask] if self.projection.bias is not None else None  # noqa: E501
            return weight, bias

    def to_static_op(self: PatchEmbed) -> nn.Module:
        """Convert dynamic PatchEmbed to static PatchEmbed."""
        self.check_if_mutables_fixed()
        assert self.mutable_embed_dims is not None

        weight, bias = self._get_dynamic_params()
        static_patch_embed = self.static_op_factory(
            img_size=self.img_size,
            in_channels=3,
            embed_dims=self.mutable_embed_dims.current_choice)

        static_patch_embed.projection.weight = nn.Parameter(weight.clone())
        static_patch_embed.projection.bias = nn.Parameter(bias.clone())

        return static_patch_embed
