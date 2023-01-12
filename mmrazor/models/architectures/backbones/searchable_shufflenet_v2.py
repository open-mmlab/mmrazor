# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import ModuleList, Sequential
from mmengine.model.weight_init import constant_init, normal_init
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.registry import MODELS

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')


@MODELS.register_module()
class SearchableShuffleNetV2(BaseBackbone):
    """Based on ShuffleNetV2 backbone.

    Args:
        arch_setting (list[list]): Architecture settings.
        stem_multiplier (int): Stem multiplier - adjusts the number of
            channels in the first layer. Default: 1.
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (4, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        with_last_layer (bool): Whether is last layer.
            Default: True, which means not need to add `Placeholder``.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        init_cfg (dict | list[dict], optional): initialization configuration
            dict to define initializer. OpenMMLab has implemented
            6 initializers, including ``Constant``, ``Xavier``, ``Normal``,
            ``Uniform``, ``Kaiming``, and ``Pretrained``.

    Examples:
        >>> mutable_cfg = dict(
        ...     type='OneShotMutableOP',
        ...     candidates=dict(
        ...         shuffle_3x3=dict(
        ...             type='ShuffleBlock',
        ...             kernel_size=3,
        ...             norm_cfg=dict(type='BN'))))
        >>> arch_setting = [
        ...     # Parameters to build layers. 3 parameters are needed to
        ...     # construct a layer, from left to right:
        ...     # channel, num_blocks, mutable cfg.
        ...     [64, 4, mutable_cfg],
        ...     [160, 4, mutable_cfg],
        ...     [320, 8, mutable_cfg],
        ...     [640, 4, mutable_cfg]
        ... ]
        >>> model = SearchableShuffleNetV2(arch_setting=arch_setting)
    """

    def __init__(self,
                 arch_setting: List[List],
                 stem_multiplier: int = 1,
                 widen_factor: float = 1.0,
                 out_indices: Sequence[int] = (4, ),
                 frozen_stages: int = -1,
                 with_last_layer: bool = True,
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Dict = dict(type='BN'),
                 act_cfg: Dict = dict(type='ReLU'),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        layers_nums = 5 if with_last_layer else 4
        for index in out_indices:
            if index not in range(0, layers_nums):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 5). But received {index}')

        self.frozen_stages = frozen_stages
        if frozen_stages not in range(-1, layers_nums):
            raise ValueError('frozen_stages must be in range(-1, 5). '
                             f'But received {frozen_stages}')

        super().__init__(init_cfg)

        self.arch_setting = arch_setting
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        last_channels = 1024
        self.in_channels = 16 * stem_multiplier
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.layers = ModuleList()
        for channel, num_blocks, mutable_cfg in arch_setting:
            out_channels = round(channel * widen_factor)
            layer = self._make_layer(out_channels, num_blocks,
                                     copy.deepcopy(mutable_cfg))
            self.layers.append(layer)

        if with_last_layer:
            self.layers.append(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=last_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def _make_layer(self, out_channels: int, num_blocks: int,
                    mutable_cfg: Dict) -> Sequential:
        """Stack mutable blocks to build a layer for ShuffleNet V2.

        Note:
            Here we use ``module_kwargs`` to pass dynamic parameters such as
            ``in_channels``, ``out_channels`` and ``stride``
            to build the mutable.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
            mutable_cfg (dict): Config of mutable.

        Returns:
            mmengine.model.Sequential: The layer made.
        """
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1

            mutable_cfg.update(
                module_kwargs=dict(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride))
            layers.append(MODELS.build(mutable_cfg))
            self.in_channels = out_channels

        return Sequential(*layers)

    def _freeze_stages(self) -> None:
        """Freeze params not to update in the specified stages."""
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self) -> None:
        """Init weights of ``SearchableShuffleNetV2``."""
        super().init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv1' in name:
                    normal_init(m, mean=0, std=0.01)
                else:
                    normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, val=1, bias=0.0001)
                if isinstance(m, _BatchNorm):
                    if m.running_mean is not None:
                        nn.init.constant_(m.running_mean, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward computation.

        Args:
            x (tensor): x contains input data for forward computation.
        """
        x = self.conv1(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Set module status before forward computation."""
        super().train(mode)

        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
