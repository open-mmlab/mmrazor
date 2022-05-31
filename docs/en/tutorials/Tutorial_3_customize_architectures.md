# Toturial 3: Customize Architectures

Different from other tasks, architecture may consist of some searchable model components in NAS. In MMRazor, you can not only develop some common model components like other codebases of OpenMMLab, but also develop some searchable model components. Here is how to develop searchable model components and common model components.

## Develop searchable model components

Here we show how to add a new searchable backbone with an example of searchable_shufflenet_v2.

1. Define a new backbone

   Create a new file `mmrazor/models/architectures/components/backbones/searchable_shufflenet_v2.py`, class `SearchableShuffleNetV2` inherits from `BaseBackBone` of mmcls, which is the codebase that you will to build the model.

   ```python
   import torch.nn as nn
   from mmcls.models.backbones.base_backbone import BaseBackbone
   from mmcls.models.builder import BACKBONES

   @BACKBONES.register_module()
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

2. Replace layers with placeholders

   ```python
   from ...utils import Placeholder

   @BACKBONES.register_module()
   class SearchableShuffleNetV2(BaseBackbone):

       ...

       def _make_layer(self, out_channels, num_blocks, stage_idx):
           """Stack blocks to make a layer.

           Args:
               out_channels (int): out_channels of the block.
               num_blocks (int): number of blocks.
           """
           layers = []
           for i in range(num_blocks):
               stride = 2 if i == 0 else 1
               layers.append(
                   Placeholder(
                       group='all_blocks',
                       space_id=f'stage_{stage_idx}_block_{i}',
                       in_channels=self.in_channels,
                       out_channels=out_channels,
                       stride=stride,
                       conv_cfg=self.conv_cfg,
                       norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg,
                       with_cp=self.with_cp,
                       init_cfg=self.init_cfg))
               self.in_channels = out_channels

           return nn.Sequential(*layers)

       ...

   ```

3. Import the module

   You can either add the following line to `mmrazor/models/architectures/components/backbones/__init__.py`

   ```python
   from .searchable_shufflenet_v2 import SearchableShuffleNetV2
   ```

   or alternatively add

   ```python
   custom_imports = dict(
       imports=['mmrazor.models.architectures.components.backbones.searchable_shufflenet_v2'],
       allow_failed_imports=False)
   ```

   to the config file to avoid modifying the original code.

4. Use the backbone in your config file

   ```python
   architecture = dict(
       type=xxx,
       model=dict(
           ...
           backbone=dict(
               type='mmcls.SearchableShuffleNetV2',
               arg1=xxx,
               arg2=xxx),
           ...
   ```

## Develop common model components

Here we show how to add a new backbone with an example of xxxNet.

1. Define a new backbone

   Create a new file `mmrazor/models/architectures/components/backbones/xxxnet.py`.

   ```python
   import torch.nn as nn
   from ..builder import BACKBONES

   @BACKBONES.register_module()
   class xxxNet(nn.Module):

       def __init__(self, arg1, arg2):
           pass

       def forward(self, x):  # should return a tuple
           pass
   ```

2. Import the module

   You can either add the following line to `mmrazor/models/architectures/components/backbones/__init__.py`

   ```python
   from .xxxnet import xxxNet
   ```

   or alternatively add

   ```python
   custom_imports = dict(
       imports=['mmrazor.models.architectures.components.backbones.xxxnet'],
       allow_failed_imports=False)
   ```

   to the config file to avoid modifying the original code.

3. Use the backbone in your config file

   ```python
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
