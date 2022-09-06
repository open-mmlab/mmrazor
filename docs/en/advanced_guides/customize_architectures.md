# Customize Architectures

Different from other tasks, architectures in MMRazor may consist of some special model components, such as **searchable backbones, connectors, dynamic ops**. In MMRazor, you can not only develop some common model components like other codebases of OpenMMLab, but also develop some special model components. Here is how to develop searchable model components and common model components.

## Develop searchable model components

1. Define a new backbone

Create a new file `mmrazor/models/architectures/backbones/searchable_shufflenet_v2.py`, class `SearchableShuffleNetV2` inherits from `BaseBackBone` of mmcls, which is the codebase that you will use to build the model.

```Python
# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn import ConvModule, constant_init, normal_init
from mmcv.runner import ModuleList, Sequential
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.registry import MODELS

@MODELS.register_module()
class SearchableShuffleNetV2(BaseBackbone):

    def __init__(self, ):
        pass

    def _make_layer(self, out_channels, num_blocks, stage_idx):
        pass

    def _freeze_stages(self):
        pass

    def init_weights(self):
        pass

    def forward(self, x):
        pass

    def train(self, mode=True):
        pass
```

2. Build the architecture of the new backbone based on `arch_setting`

```Python
@MODELS.register_module()
class SearchableShuffleNetV2(BaseBackbone):
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

            # build the first layer
            self.conv1 = ConvModule(
                in_channels=3,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            # build the middle layers
            self.layers = ModuleList()
            for channel, num_blocks, mutable_cfg in arch_setting:
                out_channels = round(channel * widen_factor)
                layer = self._make_layer(out_channels, num_blocks,
                                         copy.deepcopy(mutable_cfg))
                self.layers.append(layer)

            # build the last layer
            if with_last_layer:
                self.layers.append(
                    ConvModule(
                        in_channels=self.in_channels,
                        out_channels=last_channels,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
```

3. Implement`_make_layer` with `mutable_cfg`

```Python
@MODELS.register_module()
class SearchableShuffleNetV2(BaseBackbone):

    ...

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
            mmcv.runner.Sequential: The layer made.
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

    ...
```

4. Implement other common methods

You can refer to the implementation of `ShuffleNetV2` in mmcls for finishing other common methods.

5. Import the module

You can either add the following line to `mmrazor/models/architectures/backbones/__init__.py`

```Python
from .searchable_shufflenet_v2 import SearchableShuffleNetV2

__all__ = ['SearchableShuffleNetV2']
```

or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.architectures.backbones.searchable_shufflenet_v2'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

6. Use the backbone in your config file

```Python
architecture = dict(
    type=xxx,
    model=dict(
        ...
        backbone=dict(
            type='mmrazor.SearchableShuffleNetV2',
            arg1=xxx,
            arg2=xxx),
        ...
```

## Develop common model components

Here we show how to add a new backbone with an example of `xxxNet`.

1. Define a new backbone

Create a new file `mmrazor/models/architectures/backbones/xxxnet.py`, then implement the class `xxxNet`.

```Python
from mmengine.model import BaseModule
from mmrazor.registry import MODELS

@MODELS.register_module()
class xxxNet(BaseModule):

    def __init__(self, arg1, arg2, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        pass

    def forward(self, x):
        pass
```

2. Import the module

You can either add the following line to `mmrazor/models/architectures/backbones/__init__.py`

```Python
from .xxxnet import xxxNet

__all__ = ['xxxNet']
```

or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.architectures.backbones.xxxnet'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

3. Use the backbone in your config file

```Python
architecture = dict(
    type=xxx,
    model=dict(
        ...
        backbone=dict(
            type='xxxNet',
            arg1=xxx,
            arg2=xxx),
        ...
```

How to add other model components is similar to backbone's. For more details, please refer to other codebases' docs.
