# User Guides: Pruning Framework

## Background

// TODO

## Pruning Framework

This document introduces the pruning framework in mmrazor. Our pruning framework can help you prune a model automatically, and it is easy to extend new algorithms.

The pruning framework consists of four core modules: PruneAlgorithm, ChanelMutator, MutableChannelUnit, and DynamicOp. Their main features are detailed as below:

| Module             | Features                                                              |
| ------------------ | --------------------------------------------------------------------- |
| PruneAlgorithm     | Controls training process.                                            |
| ChanelMutator      | manages the pruning structure of the model.                           |
| MutableChannelUnit | makes pruning decisions.                                              |
| DynamicOp          | forwards with mutable number of channels, and exports pruned modules. |

<p align="center">
    <img src="./images/framework-framework.png" width="250"/>
</p>
## PruneAlgorithm

<p align="center">
    <img src="./images/framework-algorithm.png" width="400">
</p>

PruneAlgorithm inherents from BaseAlgorithm. It controls the training process, like deciding when to prune the model in the training/finetune process.

For example, ItePruneAlgorithm prunes the model iteratively by certain epochs.

Here is an example of how to use PruneAlgoritm.

```python
from mmrazor.models.algorithms import ItePruneAlgorithm
from mmengine.model import BaseModel
import torch.nn as nn

class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)

model = Model()
algorithm = ItePruneAlgorithm(model,
                              mutator_cfg=dict(
                                  type='BaseChannelMutator',
                                  channl_unit_cfg=dict(type='L1ChannelUnit')),)
print(algorithm)
# ItePruneAlgorithm(
#   (data_preprocessor): BaseDataPreprocessor()
#   (architecture): Model(
#     (data_preprocessor): BaseDataPreprocessor()
#     (conv): DynamicConv2d(
#       3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
#       (mutable_attrs): ModuleDict(
#         (in_channels): MutableChannelContainer(name=, num_channels=3, activated_channels: 3
#         (out_channels): MutableChannelContainer(name=, num_channels=8, activated_channels: 8
#       )
#     )
#   )
#   (mutator): BaseChannelMutator()
# )

```

## ChanelMutator

<p align="center">
    <img src="./images/framework-ChanelGroupMutator.png" width="500">
</p>

ChanelMutator controls the pruning structure of the model. In other words, ChanelMutator decides how many channels that each layer prunes. Usually, given a pruning target, such as flops, latency or pruning ratio, ChannelUnitMutator will output a pruning structure for the model. The pruning structure is variable. The default definition is channel remaining ratio, and it's also easy to extend to the number of channels or channel buckets.

As the channels of some layers are related with each other, the related layers share one pruning decision. We put these related layers to a MutableChannelUnit. Therefore, the ChanelMutator directly decides the pruning ratio of each MutableChannelUnit.

```python
from mmrazor.models.mutators import BaseChannelMutator
from mmengine.model import BaseModel
import torch.nn as nn

class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.Conv2d(8, 16, 3, 2, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 1000)

    def forward(self, x):
        x_ = self.pool(self.feature(x))
        return self.head(x_.flatten(1))

model = Model()
mutator = BaseChannelMutator()
mutator.prepare_from_supernet(model)
print(mutator.sample_choices())
# {
#     'feature.0_(0, 8)_out_1_in_1': 0.5,
#     'feature.1_(0, 16)_out_1_in_1': 0.5625
# }
```

## MutableChannelUnit

<p align="center">
    <img src="./images/unit.png"  width="700">
</p>

Because some layers' channels are related, the related layers are collected and put in a MutableChannelUnit.

Each MutableChannelUnit accepts a pruning ratio and generates a channel mask for all related layers.

All related layers are divided into two types: output_related and input_related.

1. The output channels of output-related layers are in the MutableChannelUnit.
2. The input channels of input-related layers are in the MutableChannelUnit.

Besides, basic PyTorch modules are converted to DynamicOps, which can deal with a mutable number of channels.

## DynamicOP

<p align="center">
    <img src="./images/framework-op.png" width="300">
</p>

Dynamic-ops inherit from basic torch module, like nn.Conv2d or nn.Linear. They are able to forward with mutable number of channels, and export pruned torch modules. Compared with basic torch modules, each DynamicOp has two BaseMutableChannel modules, which control the input-channels and output-channels respectively.

## More Documents about Pruning

Please refer to the following doocuments for more details.

- Development tutorials
  - [How to prune your model](./tutorials/how_to_prune_your_model.md)
  - [How to use config tool of pruning](./tutorials/how_to_use_config_tool_of_pruning.md)
- READMEs
  - [ChannelMutator](./READMEs/channel_mutator.ipynb)
  - [MutableChannelUnit](./READMEs/mutable_channel_unit.ipynb)
