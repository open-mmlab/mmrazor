# L1-norm pruning

> [Pruning Filters for Efficient ConvNets.](https://arxiv.org/pdf/1608.08710.pdf)

<!-- [ALGORITHM] -->

## Implementation

L1-norm pruning is a classical filter pruning algorithm. It prunes filers(channels) according to the l1-norm of the weight of a conv layer.

We use ItePruneAlgorithm and L1MutableChannelUnit to implement l1-norm pruning. Please refer to [Pruning User Guide](../../../../docs/en/user_guides/pruning_user_guide.md) for more configuration detail.

| Model             | Top-1 | Gap   | Flop(G) | Pruned | Parameters | Pruned | Config                                                                                                 | Download                                                                                                                                                                                                                                                |
| ----------------- | ----- | ----- | ------- | ------ | ---------- | ------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ResNet34          | 73.62 | -     | 3.68    | -      | 2.18       | -      | [mmcls](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/resnet34_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.log.json)                         |
| ResNet34_Pruned_A | 73.61 | -0.01 | 3.10    | 15.8%  | 2.01       | 7.8%   | [config](./l1-norm_resnet34_8xb32_in1k_a.py)                                                           | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/l1-norm/l1-norm_resnet34_8xb32_in1k_a.pth) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/l1-norm/l1-norm_resnet34_8xb32_in1k_a.json) |
| ResNet34_Pruned_B | 73.20 | -0.42 | 2.79    | 24.2%  | 1.95       | 10.6%  | [config](./l1-norm_resnet34_8xb32_in1k_a.py)                                                           | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/l1-norm/l1-norm_resnet34_8xb32_in1k_b.pth) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/l1-norm/l1-norm_resnet34_8xb32_in1k_b.json) |
| ResNet34_Pruned_C | 73.89 | +0.27 | 3.40    | 7.6%   | 2.02       | 7.3%   | [config](./l1-norm_resnet34_8xb32_in1k_a.py)                                                           | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/l1-norm/l1-norm_resnet34_8xb32_in1k_c.pth) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/pruning/l1-norm/l1-norm_resnet34_8xb32_in1k_c.json) |

**Note:** There is a different implementation from the original paper. We pruned the layers related to the shortcut with a shared pruning decision, while the original paper pruned them separately in *Pruned C*. This may be why our *Pruned C* outperforms *Prune A* and *Prune B*, while *Pruned C* is worst in the original paper.
