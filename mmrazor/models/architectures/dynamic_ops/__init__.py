# Copyright (c) OpenMMLab. All rights reserved.
from .bricks.dynamic_container import DynamicSequential
from .bricks.dynamic_conv import BigNasConv2d, DynamicConv2d, OFAConv2d
from .bricks.dynamic_embed import DynamicPatchEmbed
from .bricks.dynamic_linear import DynamicLinear
from .bricks.dynamic_norm import (DynamicBatchNorm1d, DynamicBatchNorm2d,
                                  DynamicBatchNorm3d, DynamicLayerNorm,
                                  SwitchableBatchNorm2d)
from .bricks.dynamic_relative_position import DynamicRelativePosition2D
from .bricks.dynamic_attention import DynamicMultiheadAttention
from .mixins.dynamic_conv_mixins import DynamicConvMixin
from .mixins.dynamic_mixins import (DynamicBatchNormMixin, DynamicChannelMixin,
                                    DynamicLinearMixin, DynamicMixin)
from .mixins.dynamic_rp_mixins import DynamicRelativePosition2DMixin
from .mixins.dynamic_mha_mixins import DynamicMHAMixin
from .head import DynamicLinearClsHead

__all__ = [
    'BigNasConv2d',
    'DynamicConv2d',
    'OFAConv2d',
    'DynamicLinear',
    'DynamicBatchNorm1d',
    'DynamicBatchNorm2d',
    'DynamicBatchNorm3d',
    'DynamicMixin',
    'DynamicChannelMixin',
    'DynamicBatchNormMixin',
    'DynamicLinearMixin',
    'SwitchableBatchNorm2d',
    'DynamicConvMixin',
    # Autoformer
    'DynamicSequential',
    'DynamicPatchEmbed',
    'DynamicLayerNorm',
    'DynamicRelativePosition2D',
    'DynamicRelativePosition2DMixin',
    'DynamicMultiheadAttention',
    'DynamicMHAMixin',
    'DynamicLinearClsHead'
]
