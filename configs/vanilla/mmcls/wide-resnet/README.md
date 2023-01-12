# Wide-ResNet

> [Wide Residual Networks](https://arxiv.org/abs/1605.07146)

<!-- [ALGORITHM] -->

## Abstract

Deep residual networks were shown to be able to scale up to thousands of layers and still have improving performance. However, each fraction of a percent of improved accuracy costs nearly doubling the number of layers, and so training very deep residual networks has a problem of diminishing feature reuse, which makes these networks very slow to train. To tackle these problems, in this paper we conduct a detailed experimental study on the architecture of ResNet blocks, based on which we propose a novel architecture where we decrease depth and increase width of residual networks. We call the resulting network structures wide residual networks (WRNs) and show that these are far superior over their commonly used thin and very deep counterparts. For example, we demonstrate that even a simple 16-layer-deep wide residual network outperforms in accuracy and efficiency all previous deep residual networks, including thousand-layer-deep networks, achieving new state-of-the-art results on CIFAR, SVHN, COCO, and significant improvements on ImageNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/156701329-2c7ec7bc-23da-401b-86bf-dea8567ccee8.png" width="90%"/>
</div>

## Results and models

### Cifar10

| Model  | Top-1 (%) |                Config                 |                                                                                                                                       Download                                                                                                                                        |
| :----: | :-------: | :-----------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| WRN-16 |   93.04   | [config](./wrn16-w2_b16x8_cifar10.py) |   [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/wide_resnet/wrn16_2_b16x8_cifar10_20220831_204709-446b466e.pth) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/wide_resnet/wrn16_2_b16x8_cifar10_20220831_204709-446b466e.json)   |
| WRN-22 |  94.8700  | [config](./wrn22-w4_b16x8_cifar10.py) | [model](https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn22-w4_b16x8_cifar10/wrn22-w4_b16x8_cifar10_20221201_170638-1d044c6f.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn22-w4_b16x8_cifar10/wrn22-w4_b16x8_cifar10_20221201_170638-1d044c6f.json) |
| WRN-28 |   95.41   | [config](./wrn28-w4_b16x8_cifar10.py) |   [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/wide_resnet/wrn28_4_b16x8_cifar10_20220831_173536-d6f8725c.pth) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/wide_resnet/wrn28_4_b16x8_cifar10_20220831_173536-d6f8725c.json)   |
| WRN-40 |  94.6700  | [config](./wrn40-w2_b16x8_cifar10.py) | [model](https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn40-w2_b16x8_cifar10/wrn40-w2_b16x8_cifar10_20221201_170318-761c8c55.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn40-w2_b16x8_cifar10/wrn40-w2_b16x8_cifar10_20221201_170318-761c8c55.json) |

## Citation

```bibtex
@INPROCEEDINGS{Zagoruyko2016WRN,
    author = {Sergey Zagoruyko and Nikos Komodakis},
    title = {Wide Residual Networks},
    booktitle = {BMVC},
    year = {2016}}
```
