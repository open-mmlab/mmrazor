# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmcv.cnn import ConvModule
from mmengine.model import Sequential
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.registry import MODELS

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
    from mmcls.models.utils import make_divisible
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')
    make_divisible = get_placeholder('mmcls')


@MODELS.register_module()
class SearchableMobileNetV2(BaseBackbone):
    """Searchable MobileNetV2 backbone.

    Args:
        arch_setting (list[list]): Architecture settings.
        first_channels (int): Channel width of first ConvModule. Default: 32.
        last_channels (int): Channel width of last ConvModule. Default: 1200.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        init_cfg (dict | list[dict], optional): initialization configuration
            dict to define initializer. OpenMMLab has implemented
            6 initializers, including ``Constant``, ``Xavier``, ``Normal``,
            ``Uniform``, ``Kaiming``, and ``Pretrained``.

    Excamples:
        >>> mutable_cfg = dict(
        ...     type='OneShotMutableOP',
        ...     candidates=dict(
        ...         mb_k3e1=dict(
        ...             type='MBBlock',
        ...             kernel_size=3,
        ...             expand_ratio=1,
        ...             norm_cfg=dict(type='BN'),
        ...             act_cfg=dict(type='ReLU6'))))
        >>> arch_setting = [
        ...     # Parameters to build layers. 4 parameters are needed to
        ...     # construct a layer, from left to right:
        ...     # channel, num_blocks, stride, mutable cfg.
        ...     [16, 1, 1, mutable_cfg],
        ...     [24, 2, 2, mutable_cfg],
        ...     [32, 3, 2, mutable_cfg],
        ...     [64, 4, 2, mutable_cfg],
        ...     [96, 3, 1, mutable_cfg],
        ...     [160, 3, 2, mutable_cfg],
        ...     [320, 1, 1, mutable_cfg]
        ... ]
        >>> model = SearchableMobileNetV2(arch_setting=arch_setting)
    """

    def __init__(
        self,
        arch_setting: List[List],
        first_channels: int = 32,
        last_channels: int = 1280,
        widen_factor: float = 1.,
        out_indices: Sequence[int] = (7, ),
        frozen_stages: int = -1,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Dict = dict(type='BN'),
        act_cfg: Dict = dict(type='ReLU6'),
        norm_eval: bool = False,
        with_cp: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Kaiming', layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]
    ) -> None:
        for index in out_indices:
            if index not in range(8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')

        super().__init__(init_cfg)

        self.arch_setting = arch_setting
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(first_channels * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(arch_setting):
            channel, num_blocks, stride, mutable_cfg = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self._make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                mutable_cfg=copy.deepcopy(mutable_cfg))
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(last_channels * widen_factor)
        else:
            self.out_channel = last_channels

        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.add_module('conv2', layer)
        self.layers.append('conv2')

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int,
                    mutable_cfg: Dict) -> Sequential:
        """Stack mutable blocks to build a layer for SearchableMobileNetV2.

        Note:
            Here we use ``module_kwargs`` to pass dynamic parameters such as
            ``in_channels``, ``out_channels`` and ``stride``
            to build the mutable.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block.
            mutable_cfg (dict): Config of mutable.

        Returns:
            mmengine.model.Sequential: The layer made.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1

            mutable_cfg.update(
                module_kwargs=dict(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride))
            layers.append(MODELS.build(mutable_cfg))

            self.in_channels = out_channels

        return Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward computation.

        Args:
            x (tensor): x contains input data for forward computation.
        """
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self) -> None:
        """Freeze params not to update in the specified stages."""
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        """Set module status before forward computation."""
        super().train(mode)

        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
