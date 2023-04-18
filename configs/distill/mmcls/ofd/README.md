# Overhaul

> [A Comprehensive Overhaul of Feature Distillation](https://arxiv.org/abs/1904.01866)

<!-- [ALGORITHM] -->

## Abstract

We investigate the design aspects of feature distillation methods achieving network compression and propose a novel feature distillation method in which the distillation loss is designed to make a synergy among various aspects: teacher transform, student transform, distillation feature position and distance function. Our proposed distillation loss includes a feature transform with a newly designed margin ReLU, a new distillation feature position, and a partial L2 distance function to skip redundant information giving adverse effects to the compression of student. In ImageNet, our proposed method achieves 21.65% of top-1 error with ResNet50, which outperforms the performance of the teacher network, ResNet152. Our proposed method is evaluated on various tasks such as image classification, object detection and semantic segmentation and achieves a significant performance improvement in all tasks. The code is available at [link](https://sites.google.com/view/byeongho-heo/overhaul)

### Feature-based Distillation

![feature_base](https://user-images.githubusercontent.com/88702197/187423965-bb3bde16-c71a-43c6-903c-69aff1005415.png)

### Margin ReLU

![margin_relu](https://user-images.githubusercontent.com/88702197/187423981-67106ac2-48d9-4002-8b32-b92a90b1dacd.png)

## Results and models

### 1. Classification

#### Vanilla

| Dataset | Model                                                                   | Top-1 (%) | Download                                                                                                                                                                                                                              |
| ------- | ----------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR10 | [WRN16-2](../../../vanilla/mmcls/wide-resnet/wrn16-w2_b16x8_cifar10.py) | 93.43     | [model](https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn16_2_b16x8_cifar10_20220831_204709-446b466e.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn16_2_b16x8_cifar10_20220831_204709-446b466e.json) |
| CIFAR10 | [WRN28-4](../../../vanilla/mmcls/wide-resnet/wrn28-w4_b16x8_cifar10.py) | 95.49     | [model](https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn28_4_b16x8_cifar10_20220831_173536-d6f8725c.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn28_4_b16x8_cifar10_20220831_173536-d6f8725c.json) |

#### Distillation

| Dataset | Model   | Flops(M) | Teacher | Top-1 (%) | Configs                                                     | Download                                                                                                                                                                                                                                                                     |
| ------- | ------- | -------- | ------- | --------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR10 | WRN16-2 | 101      | WRN28-4 | 94.21     | [config](./ofd_backbone_resnet50_resnet18_8xb16_cifar10.py) | [model](https://download.openmmlab.com/mmrazor/v1/overhaul/ofd_backbone_resnet50_resnet18_8xb16_cifar10_20230417_192216-ace2908f.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/overhaul/ofd_backbone_resnet50_resnet18_8xb16_cifar10_20230417_192216-ace2908f.log) |

## Getting Started

### Distillation training.

```bash
sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/ofd/ofd_backbone_resnet50_resnet18_8xb16_cifar10.py \
  $DISTILLATION_WORK_DIR
```

### Test

```bash
sh tools/slurm_test.sh $PARTITION $JOB_NAME \
  configs/distill/mmcls/ofd/ofd_backbone_resnet50_resnet18_8xb16_cifar10.py \
  $DISTILLATION_WORK_DIR/latest.pth --eval $EVAL_SETTING
```

## Citation

```latex
@inproceedings{heo2019overhaul,
  title={A Comprehensive Overhaul of Feature Distillation},
  author={Heo, Byeongho and Kim, Jeesoo and Yun, Sangdoo and Park, Hyojin and Kwak, Nojun and Choi, Jin Young},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
