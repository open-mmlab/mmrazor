# A Comprehensive Overhaul of Feature Distillation

## Abstract

We investigate the design aspects of feature distillation methods achieving network compression and propose a novel feature distillation method in which the distillation loss is designed to make a synergy among various aspects: teacher transform, student transform, distillation feature position and distance function. Our proposed distillation loss includes a feature transform with a newly designed margin ReLU, a new distillation feature position, and a partial L2 distance function to skip redundant information giving adverse effects to the compression of student. In ImageNet, our proposed method achieves 21.65% of top-1 error with ResNet50, which outperforms the performance of the teacher network, ResNet152. Our proposed method is evaluated on various tasks such as image classification, object detection and semantic segmentation and achieves a significant performance improvement in all tasks. The code is available at [link](https://sites.google.com/view/byeongho-heo/overhaul)

### Feature-based Distillation

![structure](../../../docs/imgs/model_zoo/overhaul/feature_base.png)

### Margin ReLU

![margin_relu](../../../docs/imgs/model_zoo/overhaul/margin_relu.png)

## Citation

```latex
@inproceedings{heo2019overhaul,
  title={A Comprehensive Overhaul of Feature Distillation},
  author={Heo, Byeongho and Kim, Jeesoo and Yun, Sangdoo and Park, Hyojin and Kwak, Nojun and Choi, Jin Young},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## Results and models

### 1. Classification

#### Vanilla

| Dataset  | Model      | Top-1 (%) | Top-5 (%) | Download                                                                                                                                                                                                                          |
| -------- | ---------- | --------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR10  | WRN16-2    | 93.43     | 99.75     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/9559a073-185d-490e-a095-82c2c96486d8)                                                                                                                       |
| CIFAR10  | WRN28-4    | 95.49     | 99.81     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/5fe74079-dd09-49ae-b2aa-ba29e62efe02)                                                                                                                       |
| CIFAR10  | ResNet-18  | 94.96     | 99.81     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/ba8acecf-1ee2-40a3-a48d-5ebd3e88d867)                                                                                                                       |
| CIFAR10  | ResNet-50  | 95.63     | 99.85     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/6d776c9e-8b3d-4361-9b41-3874ca27259d)                                                                                                                       |
| CIFAR100 | WRN16-2    | 73.5      | 92.67     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/c35416eb-6ccc-4918-9da4-b282afcbcbb6)                                                                                                                       |
| CIFAR100 | WRN16-4    | 78.12     | 93.91     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/2e18ef9e-8ed6-41dd-a8b7-94b017a262a8)                                                                                                                       |
| CIFAR100 | WRN28-4    | 79.2      | 94.37     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/324c6cf1-5b72-4e82-9f4a-f7ab3a542f68)                                                                                                                       |
| CIFAR100 | ResNet-18  | 78.89     | 94.21     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/4bdd3596-f709-4122-9755-9553fdfc0f60)                                                                                                                       |
| CIFAR100 | ResNet-50  | 79.98     | 94.91     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/4bdd3596-f709-4122-9755-9553fdfc0f60)                                                                                                                       |
| ImageNet | ResNet-18  | 69.90     | 89.43     | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.log.json)   |
| ImageNet | ResNet-50  | 76.55     | 93.06     | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.log.json)   |
| ImageNet | ResNet-152 | 78.48     | 94.06     | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.log.json) |

#### Distillation

| Dataset  | Model     | Flops(M) | Teacher    | Top-1 (%) | Top-5 (%) | Configs                                                                                                                  | Download                                                                                             | Remarks        |
| -------- | --------- | -------- | ---------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | -------------- |
| CIFAR10  | WRN16-2   | 101      | WRN28-4    | 95.23     | 99.79     | [config](/configs/kd/ofd/classification/wideresnet/ofdloss_wrn16_2_cifar10_distillation_8xb16_teacher_wrn28_4_train.py)  | [model & log](https://autolink.sensetime.com/pages/model/share/c004a590-e5ca-4b9c-934e-8fc67d5b6f81) | GML classified |
| CIFAR10  | ResNet-18 | 555      | ResNet-50  | 95.41     | 99.86     | [config](/configs/kd/ofd/classification/resnet/ofdloss_res18_cifar10_distillation_8xb16_teacher_res50_train.py)          | [model & log](https://autolink.sensetime.com/pages/model/share/9820af40-f84d-4a4e-b9e0-916b28817de9) | GML classified |
| CIFAR100 | WRN16-2   | 101      | WRN28-4    | 74.32     | 93.34     | [config](/configs/kd/ofd/classification/wideresnet/ofdloss_wrn16_2_cifar100_distillation_8xb16_teacher_wrn28_4_train.py) | [model & log](https://autolink.sensetime.com/pages/model/share/38659fcd-3b5b-4633-8bcf-5a2213969fbf) | GML classified |
| CIFAR100 | WRN16-4   | 555      | WRN28-4    | 79.25     | 94.85     | [config](/configs/kd/ofd/classification/wideresnet/ofdloss_wrn16_4_cifar100_distillation_8xb16_teacher_wrn28_4_train.py) | [model & log](https://autolink.sensetime.com/pages/model/share/d2849e02-e7ce-4692-b94f-f1564a192bbe) | GML classified |
| CIFAR100 | ResNet-18 | 555      | ResNet-50  | 80.02     | 95.26     | [config](/configs/kd/ofd/classification/resnet/ofdloss_res18_cifar100_distillation_8xb16_teacher_res50_train.py)         | [model & log](https://autolink.sensetime.com/pages/model/share/f81ffb1a-0192-40aa-b040-d91628c9f458) | GML classified |
| ImageNet | ResNet-18 | 1814     | ResNet-50  | 70.34     | 90.08     | [config](/configs/kd/ofd/classification/resnet/ofdloss_res18_imagenet_distillation_8xb32_teacher_res50_train.py)         | [model & log](https://autolink.sensetime.com/pages/model/share/07b46bfb-1551-496d-be36-7efdba5ccf25) | GML classified |
| ImageNet | ResNet-50 | 4089     | ResNet-152 | 78.054    | 94.194    | [config](/configs/kd/ofd/classification/resnet/ofdloss_res50_imagenet_distillation_8xb32_teacher_res152_train.py)        | [model & log](https://autolink.sensetime.com/pages/model/share/7bad50f3-64a1-458f-a906-a50d1eeb5b31) | GML classified |

### 2. Detection

#### Vanilla

| Dataset | Model       | Backbone  | bbox mAP |                                                                                                                                                  Download                                                                                                                                                   |
| ------- | ----------- | --------- | :------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| COCO    | Faster-RCNN | R-50-FPN  |   37.4   |   [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)\|[log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json)   |
| COCO    | Faster-RCNN | R-101-FPN |   39.4   | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth)\|[log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130_204655.log.json) |
| COCO    | RetinaNet   | R-50-FPN  |   36.5   |         [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth)\|[log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130_002941.log.json)         |
| COCO    | RetinaNet   | R-101-FPN |   38.5   |       [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth)\|[log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130_003055.log.json)       |

#### Distillation

| Dataset | Model      | Backbone | Teacher Backbone | Stage    | MAP  | Config                                                                                                             | Download                                                                                                    |
| ------- | ---------- | -------- | :--------------- | -------- | ---- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| COCO    | FasterRCNN | R-50-FPN | R-101-FPN        | Backbone | 38.1 | [config](/configs/kd/ofd/detection/faster_rcnn/ofdloss_faster_rcnn_r50_fpn_1x_coco_teacher_r101_backbone_mimic.py) | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/d146f02b-19b0-4f6b-be60-5db17866acc6) |
| COCO    | RetinaNet  | R-50-FPN | R-101-FPN        | Backbone | 37.3 | [config](/configs/kd/ofd/detection/retinanet/ofdloss_retinanet_r50_fpn_1x_coco_teacher_r101_backbone_mimic.py)     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/eacd59fb-f2a3-4c47-a83b-7b9fb1041eee) |

### 3. Segmentation

#### Vanilla

| Dataset    | Model      | Backbone | Lr schd | mIoU  |                                                                                                                                                                                             Download                                                                                                                                                                                             |
| ---------- | ---------- | -------- | ------- | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| VOC2012Aug | DeepLabV3+ | R-18-D8  | 40k     | 72.62 |                                                                                                                                           [model & log](http://autolink.parrots.sensetime.com/pages/model/share/c0be5c03-2f4f-461e-a67d-f49f7295b653)                                                                                                                                            |
| VOC2012Aug | DeepLabV3+ | R-101-D8 | 40k     | 78.62 |       [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth)\|[log](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333.log.json)       |
| Cityscapes | DeepLabV3+ | R-18-D8  | 80k     | 76.89 |   [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth)\|[log](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes-20201226_080942.log.json)   |
| Cityscapes | DeepLabV3+ | R-101-D8 | 80k     | 80.97 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth)\|[log](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143.log.json) |

#### Distillation

| Dataset    | Model      | Backbone | Teacher Backbone | Stage    | mIoU  | Config                                                                                                                                                      | Download                                                                                                    |
| ---------- | ---------- | -------- | :--------------- | -------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| VOC2012Aug | DeepLabV3+ | R-18-D8  | R-101-D8         | Backbone | 72.49 | [config](/configs/kd/ofd/segmentation/deeplabv3plus/ofdloss_deeplabv3plus_r18-d8_512x512_40k_voc12_distillation_8xb16_teacher_r101_backbone_mimic.py)       | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/6dfb210f-3930-4cf5-9855-e16940b8a280) |
| VOC2012Aug | DeepLabV3+ | R-18-D8  | R-101-D8         | Head     | 72.96 | [config](/configs/kd/ofd/segmentation/deeplabv3plus/ofdloss_deeplabv3plus_r18-d8_512x512_40k_voc12_distillation_8xb16_teacher_r101_head_mimic.py)           | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/615ae3ba-b2b9-4ba3-98b6-bfce2faca8f0) |
| Cityscapes | DeepLabV3+ | R-18-D8  | R-101-D8         | Backbone | 77.02 | [config](/configs/kd/ofd/segmentation/deeplabv3plus/ofdloss_deeplabv3plus_r18-d8_512x1024_80k_cityscapes_distillation_8xb16_teacher_r101_backbone_mimic.py) | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/4767303d-89d3-4ed0-aa04-acd36b7cadc9) |
| Cityscapes | DeepLabV3+ | R-18-D8  | R-101-D8         | Head     | 77.65 | [config](/configs/kd/ofd/segmentation/deeplabv3plus/ofdloss_deeplabv3plus_r18-d8_512x1024_80k_cityscapes_distillation_8xb16_teacher_r101_head_mimic.py)     | [model & log](http://autolink.parrots.sensetime.com/pages/model/share/c51aef0f-fd55-40f7-b6ea-93676e80febe) |

## Getting Started

### Modify Distillation training config

open file './configs/kd/ofd/classification/resnet/ofdloss_res18_cifar10_distillation_8xb16_teacher_res50_train.py'

```python
# TODO
name='ResNet50',
ckpt_path='work_dirs/resnet50_b16x8_cifar10_best.pth',
```

### Distillation training.

```bash
sh tools/slurm_train.sh $PARTITION $JOB_NAME \
  configs/kd/ofd/classification/resnet/ofdloss_res18_cifar10_distillation_8xb16_teacher_res50_train.py \
  $DISTILLATION_WORK_DIR
```

### Test

```bash
sh tools/slurm_test.sh $PARTITION $JOB_NAME \
  configs/kd/ofd/classification/resnet/ofdloss_res18_cifar10_distillation_8xb16_teacher_res50_train.py \
  $DISTILLATION_WORK_DIR/latest.pth --eval $EVAL_SETTING
```

## Guide to Practice

### Notes

OFD是一种feature based的蒸馏，针对relu进行设计。
OFD使用在 relu之前，通常为bn。
OFD会忽略无效的知识，比如对于一个特征 student_feature \< teacher_feautre \< 0
那么这个知识是无需被转移的

### Adapt To Various Down-stream Tasks

OFD可以运用到任意下游任务，使用方式与fitnet类似
OFD可以与response based蒸馏配合使用

### Usage

Overhaul 目前实现 只能在bn层进行蒸馏

```python
  dict(
      student_module='xxx.xxx.xxx.bn',
      teacher_module='xxx.xxx.xxx.bn',
      head=dict(
          type='OFDDistillationHead',
          loss=dict(
              type='OFDLoss',
              loss_weight=1,
          ),
          name='xxx',
          stu_connector=dict(
              type='BNConnector',
              in_channel=...,
              out_channel=...,
          ),
      )),
```
