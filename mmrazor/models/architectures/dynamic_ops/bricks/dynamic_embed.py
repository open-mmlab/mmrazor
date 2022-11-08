# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.utils import PatchEmbed
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from ..mixins.dynamic_patchembed_mixins import DynamicPatchEmbedMixin


@MODELS.register_module()
class DynamicPatchEmbed(PatchEmbed, DynamicPatchEmbedMixin):
    """Dynamic Patch Embedding.

    Args:
        img_size (int, optional): The size of input image.
            Defaults to 224.
        in_channels (int, optional): The input channel of PatchEmbed.
            Defaults to 3.
        embed_dims ([type], optional): The embedding dimensions.
            Defaults to None.
        convcfg ([type], optional): The config for convolution layers.
            Defaults to None.
    """
    accpeted_mutables = {'embed_dims'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @property
    def in_channels(self):
        return getattr(self, self.attr_mappings['in_channels'])

    @property
    def out_channels(self):
        return getattr(self, self.attr_mappings['out_channels'])

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return PatchEmbed

    @classmethod
    def convert_from(cls, module) -> nn.Module:
        """Convert a PatchEmbed to a DynamicPatchEmbed."""

        dynamic_patch_embed = cls(
            img_size=module.img_size,
            in_channels=3,
            embed_dims=module.embed_dims,
            norm_cfg=None,
            conv_cfg=None,
            init_cfg=None)

        return dynamic_patch_embed

    def forward(self, x: Tensor) -> Tensor:
        """Forward of dynamic patch embed."""
        weight, bias = self._get_dynamic_params()
        x = F.conv2d(
            x,
            weight,
            bias,
            stride=16,
            padding=self.projection.padding,
            dilation=self.projection.dilation).flatten(2).transpose(1, 2)

        return x
