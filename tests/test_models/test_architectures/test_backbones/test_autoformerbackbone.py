# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.architectures.dynamic_ops import (
    DynamicLinear, DynamicMultiheadAttention, DynamicPatchEmbed,
    DynamicRelativePosition2D, DynamicSequential)
from mmrazor.models.mutables import MutableChannelContainer
from mmrazor.registry import MODELS

arch_setting = dict(
    mlp_ratios=[3.0, 3.5, 4.0],
    num_heads=[8, 9, 10],
    depth=[14, 15, 16],
    embed_dims=[528, 576, 624])

BACKBONE_CFG = dict(
    type='mmrazor.AutoformerBackbone',
    arch_setting=arch_setting,
    img_size=224,
    patch_size=16,
    in_channels=3,
    norm_cfg=dict(type='mmrazor.DynamicLayerNorm'),
    act_cfg=dict(type='GELU'))


def test_searchable_autoformer_mutable() -> None:
    backbone = MODELS.build(BACKBONE_CFG)

    num_heads = backbone.arch_setting['num_heads']
    mlp_ratios = backbone.arch_setting['mlp_ratios']
    depth = backbone.arch_setting['depth']
    embed_dims = backbone.arch_setting['embed_dims']
    embed_dims_expansion = [i * j for i in mlp_ratios for j in embed_dims]
    head_expansion = [i * 64 for i in num_heads]

    for name, module in backbone.named_modules():
        if isinstance(module, DynamicRelativePosition2D):
            assert len(module.mutable_head_dims.current_choice) == 64
        elif isinstance(module, DynamicMultiheadAttention):
            assert len(
                module.mutable_embed_dims.current_choice) == max(embed_dims)
            assert len(module.mutable_q_embed_dims.current_choice) == max(
                head_expansion)
            assert module.mutable_num_heads.choices == num_heads
        elif isinstance(module, DynamicLinear):
            if 'fc1' in name:
                assert module.mutable_attrs['in_features'].num_channels == max(
                    embed_dims)
                assert module.mutable_attrs[
                    'out_features'].num_channels == max(embed_dims_expansion)
            elif 'fc2' in name:
                assert module.mutable_attrs['in_features'].num_channels == max(
                    embed_dims_expansion)
                assert module.mutable_attrs[
                    'out_features'].num_channels == max(embed_dims)
        elif isinstance(module, DynamicPatchEmbed):
            assert type(module.mutable_embed_dims) == MutableChannelContainer
            assert len(
                module.mutable_embed_dims.current_choice) == max(embed_dims)
        elif isinstance(module, DynamicSequential):
            assert module.mutable_depth.choices == depth
    assert backbone.last_mutable.num_channels == max(embed_dims)
