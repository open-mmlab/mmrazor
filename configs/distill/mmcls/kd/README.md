# KD

> [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

<!-- [ALGORITHM] -->

## Abstract

A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.

![pipeline](https://user-images.githubusercontent.com/88702197/187423762-e932dd3e-16cb-4714-a85f-cddfc906c1b7.png)

## Results and models

### Classification

| Location | Dataset  |                                                   Teacher                                                    |                                                   Student                                                    |  Acc  | Acc(T) | Acc(S) |                          Config                           | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| :------: | :------: | :----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  logits  | ImageNet | [resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb32_in1k.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | 71.54 | 73.62  | 69.90  | [config](./wsld_cls_head_resnet34_resnet18_8xb32_in1k.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \|[model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/distill/wsld/wsld_cls_head_resnet34_resnet18_8xb32_in1k/wsld_cls_head_resnet34_resnet18_8xb32_in1k_acc-71.54_20211222-91f28cf6.pth?versionId=CAEQHxiBgMC6memK7xciIGMzMDFlYTA4YzhlYTRiMTNiZWU0YTVhY2I5NjVkMjY2) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/distill/wsld/wsld_cls_head_resnet34_resnet18_8xb32_in1k/wsld_cls_head_resnet34_resnet18_8xb32_in1k_20211221_181516.log.json?versionId=CAEQHxiBgIDLmemK7xciIGNkM2FiN2Y4N2E5YjRhNDE4NDVlNmExNDczZDIxN2E5) |

## Citation

```latex
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff and others},
  journal={arXiv preprint arXiv:1503.02531},
  volume={2},
  number={7},
  year={2015}
}
```
