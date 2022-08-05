# Decoupled Knowledge Distillation

## Abstract

State-of-the-art distillation methods are mainly based on distilling deep features from intermediate layers, while the significance of logit distillation is greatly overlooked. To provide a novel viewpoint to study logit distillation, we reformulate the classical KD loss into two parts, i.e., target class knowledge distillation (TCKD) and non-target class knowledge distillation (NCKD). We empirically investigate and prove the effects of the two parts: TCKD transfers knowledge concerning the "difficulty" of training samples, while NCKD is the prominent reason why logit distillation works. More importantly, we reveal that the classical KD loss is a coupled formulation, which (1) suppresses the effectiveness of NCKD and (2) limits the flexibility to balance these two parts. To address these issues, we present Decoupled Knowledge Distillation (DKD), enabling TCKD and NCKD to play their roles more efficiently and flexibly. Compared with complex feature-based methods, our DKD achieves comparable or even better results and has better training efficiency on CIFAR-100, ImageNet, and MS-COCO datasets for image classification and object detection tasks. This paper proves the great potential of logit distillation, and we hope it will be helpful for future research. The code is available at this \[https://github.com/megvii-research/mdistiller\]

![avatar](../../../docs/imgs/model_zoo/dkd/dkd.png)

## Citation

```latex
@article{zhao2022decoupled,
  title={Decoupled Knowledge Distillation},
  author={Zhao, Borui and Cui, Quan and Song, Renjie and Qiu, Yiyu and Liang, Jiajun},
  journal={arXiv preprint arXiv:2203.08679},
  year={2022}
}
```

## Results and models

### 1. Classification

#### Vanilla

| Dataset  | Model     | Top-1 (%) | Top-5 (%) | Download                                                                                                                                                                                                                                |
| -------- | --------- | --------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR100 | WRN16-2   | 73.5      | 92.67     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/c35416eb-6ccc-4918-9da4-b282afcbcbb6)                                                                                                                             |
| CIFAR100 | WRN40-2   | 76.96     | 93.24     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/324c6cf1-5b72-4e82-9f4a-f7ab3a542f68)                                                                                                                             |
| CIFAR100 | ResNet-18 | 78.89     | 94.21     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/4bdd3596-f709-4122-9755-9553fdfc0f60)                                                                                                                             |
| CIFAR100 | ResNet-50 | 79.90     | 95.19     | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.log.json) |
| ImageNet | ResNet-18 | 69.90     | 89.43     | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.log.json)         |
| ImageNet | ResNet-34 | 76.55     | 93.06     | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.log.json)         |

#### Distillation

| Dataset  | Model     | Teacher   | Top-1 (%) | Top-5 (%) | Configs                                                                                                      | Download                                                                                                    |
| -------- | --------- | --------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| ImageNet | ResNet-18 | ResNet-34 | 71.566    | 90.326    | [config](/configs/kd/dkd/classification/resnet/dkd_res18_imagenet_distillation_8xb32_teacher_res34_train.py) | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/29284b08-ae3f-4df7-900a-9b5e0755b3d7) |

## Getting Started

Download teacher ckpt from

https://mmclassification.readthedocs.io/en/latest/papers/resnet.html

```python
# TODO
name='ResNet50',
ckpt_path='.../resnet50_b16x8_cifar100_20210528-67b58a1b.pth',
```

### Distillation training.

```bash
sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/kd/dkd/classification/resnet/dkd_res18_cifar100_distillation_8xb16_teacher_res50_mimic.py \
  $DISTILLATION_WORK_DIR
```

### Test

```bash
sh tools/slurm_test.sh $PARTITION $JOB_NAME \
  configs/kd/dkd/classification/resnet/dkd_res18_cifar100_distillation_8xb16_teacher_res50_mimic.py \
  $DISTILLATION_WORK_DIR/latest.sh --eval $EVAL_SETTING
```
