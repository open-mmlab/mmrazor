# KD

> [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

<!-- [ALGORITHM] -->

## Abstract

A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.

![pipeline](https://user-images.githubusercontent.com/88702197/187423762-e932dd3e-16cb-4714-a85f-cddfc906c1b7.png)

## Results and models

### Classification

| Location | Dataset  |                                                    Teacher                                                    |                                                              Student                                                               |  Acc  | Acc(T) | Acc(S) |                             Config                             | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| :------: | :------: | :-----------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  logits  | ImageNet | [resnet34](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/resnet/resnet34_8xb32_in1k.py) |           [resnet18](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/resnet/resnet18_8xb32_in1k.py)            | 71.81 | 73.62  | 69.90  |     [config](./kd_logits_resnet34_resnet18_8xb32_in1k.py)      | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/kd/kl_r18_w3/kd_logits_resnet34_resnet18_8xb32_in1k_w3_20221011_181115-5c6a834d.pth?versionId=CAEQThiBgID1_Me0oBgiIDE3NTk3MDgxZmU2YjRlMjVhMzg1ZTQwMmRhNmYyNGU2) \| [log](https://download.openmmlab.com/mmrazor/v1/kd/kl_r18_w3/kd_logits_resnet34_resnet18_8xb32_in1k_w3_20221011_181115-5c6a834d.json?versionId=CAEQThiBgMDx_se0oBgiIDQxNTM2MWZjZGRhNjRhZDZiZTIzY2Y0NDU3NDA4ODBl) |
|  logits  | ImageNet | [resnet50](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/resnet/resnet50_8xb32_in1k.py) |    [mobilenet-v2](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py)     | 73.56 | 76.55  | 71.86  |   [config](./kd_logits_resnet50_mobilenet-v2_8xb32_in1k.py)    | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/kd/kl_mbv2_w3t1/kd_logits_resnet50_mobilenet-v2_8xb32_in1k_20221025_212407-6ea9e2a5.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/kd/kl_mbv2_w3t1/kd_logits_resnet50_mobilenet-v2_8xb32_in1k_20221025_212407-6ea9e2a5.json)                                                                                                                                               |
|  logits  | ImageNet | [resnet50](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/resnet/resnet50_8xb32_in1k.py) | [shufflenet-v2](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | 70.87 | 76.55  | 69.55  | [config](./kd_logits_resnet50_shufflenet-v2-1x_16xb64_in1k.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/kd/kl_shuffle_w3t1/kd_logits_resnet50_shufflenet-v2-1x_16xb64_in1k_20221025_224424-5d748c1b.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/kd/kl_shuffle_w3t1/kd_logits_resnet50_shufflenet-v2-1x_16xb64_in1k_20221025_224424-5d748c1b.json)                                                                                                                               |

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
