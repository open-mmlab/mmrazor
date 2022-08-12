# CONTRASTIVE REPRESENTATION DISTILLATION

> [CONTRASTIVE REPRESENTATION DISTILLATION](https://arxiv.org/abs/1910.10699)

## Abstract

Often we wish to transfer representational knowledge from one neural network to another. Examples include distilling a large network into a smaller one, transferring knowledge from one sensory modality to a second, or ensembling a collection of models into a single estimator. Knowledge distillation, the standard approach to these problems, minimizes the KL divergence between the probabilistic outputs of a teacher and student network. We demonstrate that this objective ignores important structural knowledge of the teacher network. This motivates an alternative objective by which we train a student to capture signiﬁcantly more information in the teacher’s representation of the data. We formulate this objective as contrastive learning. Experiments demonstrate that our resulting new objective outperforms knowledge distillation and other cutting-edge distillers on a variety of knowledge transfer tasks, including single model compression, ensemble distillation, and cross-modal transfer. Our method sets a new state-of-the-art in many transfer tasks, and sometimes even outperforms the teacher network when combined with knowledge distillation.[Original code](http://github.com/HobbitLong/RepDistiller)

![pipeline](../../../../docs/en/imgs/model_zoo/crd/pipeline.jpg)

## Citation

```latex
@article{tian2019contrastive,
  title={Contrastive representation distillation},
  author={Tian, Yonglong and Krishnan, Dilip and Isola, Phillip},
  journal={arXiv preprint arXiv:1910.10699},
  year={2019}
}
```

## Results and models

### 1. Classification

#### Vanilla

| Dataset  | Model    | Top-1 (%) | Top-5 (%) | Download                                                                                                                                                                                                                      |
| -------- | -------- | --------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR10  | ResNet18 | 94.96     | 99.90     | [model](http://autolink.parrots.sensetime.com/pages/model/share/bea0668b-070e-42e5-826e-631d4fa8bf5c)\|[log](http://autolink.parrots.sensetime.com/pages/model/share/e841104e-0694-468a-adb8-75da0f412206)                    |
| CIFAR10  | ResNet50 | 95.63     | 99.88     | [model](http://autolink.parrots.sensetime.com/pages/model/share/ce7671be-88d2-40c5-9028-b1c70c0df3df)\|[log](http://autolink.parrots.sensetime.com/pages/model/share/21ee8493-b65e-4fac-b537-2b21914b55ac)                    |
| CIFAR10  | WRN16-2  | 93.76     | 99.87     | [model](http://autolink.parrots.sensetime.com/pages/model/share/58b8335a-0796-4873-bb7c-cf1f7f784654)                                                                                                                         |
| CIFAR10  | WRN22-4  | 95.46     | 99.91     | [model](http://autolink.parrots.sensetime.com/pages/model/share/7b99884b-ebe6-4aed-bfc3-d1cd5a817ff0)                                                                                                                         |
| CIFAR100 | ResNet18 | 79.09     | 94.25     | [model](http://autolink.parrots.sensetime.com/pages/model/share/e351d2fc-61f1-4c7e-ad3e-93c6c2f0d55d)\|[log](http://autolink.parrots.sensetime.com/pages/model/share/ce67a748-c6c7-46e8-a4a0-eae18a73dd6a)                    |
| CIFAR100 | ResNet50 | 79.98     | 95.18     | [mode](http://autolink.parrots.sensetime.com/pages/model/share/60054563-5d01-4b89-84fb-f0314781cc4c)\|[log](http://autolink.parrots.sensetime.com/pages/model/share/dcbc41f2-1931-41f4-ba53-c2eb4bb625c7)                     |
| CIFAR100 | WRN16-2  | 73.31     | 93.20     | [model](http://autolink.parrots.sensetime.com/pages/model/share/e94ac90b-9eb2-47f7-bde9-874ef33a864b)\|[log](http://autolink.parrots.sensetime.com/pages/model/share/6df6c5c6-3a40-4183-9652-eea8fbe1e442)                    |
| CIFAR100 | WRN22-4  | 78.65     | 94.19     | [model](http://autolink.parrots.sensetime.com/pages/model/share/b6c235b2-f046-4733-be10-f9e0a9947d7e)\|[log](http://autolink.parrots.sensetime.com/pages/model/share/eb2427cc-e78b-45f1-9106-8c9e61c79897)                    |
| ImageNet | ResNet18 | 69.90     | 89.43     | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth)\|[log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.log.json) |
| ImageNet | ResNet50 | 76.55     | 93.06     | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth)\|[log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.log.json) |

#### Distillation

| Dataset  | Model    | Teacher   | Top-1 (%) | Top-5 (%) | Config                                                                                                                              | Download                                                                                                                                                                                                   |
| -------- | -------- | --------- | --------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR10  | ResNet18 | ResNet50  | 94.62     | 99.86     | [config](configs/kd/crd_loss/classification/resnet/crdloss_res18_cifar10_distillation_8xb16_teacher_res50_dimout128_with_kdhead.py) | [model & log](https://autolink.sensetime.com/pages/model/share/c09b1eaf-323c-458d-a384-53dcb51ff5cb)                                                                                                       |
| CIFAR10  | WRN16-2  | WRN22-4   | 93.19     | 99.84     | [config](configs/kd/crd_loss/classification/wideresnet/crdloss_wrn16_2_cifar10_distillation_8xb16_teacher_wrn22_4.py)               | [model & log](https://autolink.sensetime.com/pages/model/share/59c17cca-807f-4a5f-942f-6d1b574e04b5)                                                                                                       |
| CIFAR100 | ResNet18 | ResNet50  | 79.38     | 94.99     | [config](configs/kd/crd_loss/classification/resnet/crdloss_res18_cifar100_distillation_8xb16_teacher_res50_dimout128.py)            | [model & log](https://autolink.sensetime.com/pages/model/share/96f63509-edf3-4bf6-90b1-32f875182803)                                                                                                       |
| CIFAR100 | ResNet20 | ResNet110 | 69.08     | 91.79     | [config](configs/kd/crd_loss/classification/resnet/crdloss_res20_cifar100_distillation_8xb16_teacher_res110.py)                     | [model & log](https://autolink.sensetime.com/pages/model/share/2866045a-b662-4a83-80e9-d49188c388b0)                                                                                                       |
| CIFAR100 | WRN16-2  | WRN22-4   | 75.69     | 94.06     | [config](../../../configs/kd/crd_loss/classification/wideresnet/crdloss_wrn16_2_cifar100_distillation_8xb16_teacher_wrn22_4.py)     | [model](http://autolink.parrots.sensetime.com/pages/model/share/6f616b03-50be-47bc-b125-7803c3b279f1)\|[log](http://autolink.parrots.sensetime.com/pages/model/share/b036e072-6dca-42b5-8b43-8aa92593b9f7) |
| ImageNet | ResNet18 | ResNet50  | 69.91     | 89.57     | [config](configs/kd/crd_loss/classification/resnet/crdloss_res18_imagenet_distillation_8xb32_teacher_res50_dimout_256.py)           | [model & log](https://autolink.sensetime.com/pages/model/share/b58df955-c581-477a-87af-15a501b65fdd)                                                                                                       |
| ImageNet | WRN16-2  | WRN22-4   | 70.92     | 89.98     | [config](configs/kd/crd_loss/classification/resnet/crdloss_res20_cifar100_distillation_8xb16_teacher_res110.py)                     | [model](http://autolink.parrots.sensetime.com/pages/model/share/a9ad7977-1a83-4ea9-ae1f-f10752cb90ad)\|[log](http://autolink.parrots.sensetime.com/pages/model/share/0593c8cc-b11b-44c4-bd15-19fbeb0f7bf5) |

## Acknowledgement

Shout out to @chengshuang18 for his special contribution.
