# L1-norm pruning

> [Pruning Filters for Efficient ConvNets.](https://arxiv.org/pdf/1608.08710.pdf)

<!-- [ALGORITHM] -->

## Implementation

L1-norm pruning is a classical filter pruning algorithm. It prunes filers(channels) according to the l1-norm of the weight of a conv layer.

We use ItePruneAlgorithm and L1MutableChannelUnit to implement l1-norm pruning. Please refer to xxxx for more configuration detail.
